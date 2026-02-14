import os
import json
import re
import logging
from typing import Optional, Type, Any, Dict, List
from dotenv import load_dotenv

# Pydantic imports
from pydantic import BaseModel, Field, ValidationError

# Google GenAI imports
from google import genai
from google.genai import types
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Define the ReAct template
reAct_template = '''You are a ReAct-style intelligent agent.

You must follow this reasoning pattern:
1. Think about the question
2. Reason through the problem step by step
3. Provide a clear, well-structured answer

Question: {question}

Please provide a detailed, thoughtful response.'''


class Gemini:
    def __init__(self, model: str = "gemini-3-flash-preview", temperature: float = 0.9, api_key: str = None, system_prompt: str = "You are a helpful AI assistant."):
        """
        Initialize Gemini client
        
        Args:
            model: Gemini model name (default: gemini-2.0-flash)
            api_key: Google API key for Gemini (fetches from env if None)
            system_prompt: System-level instructions for the model
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("API key cannot be empty. Provide it or set GEMINI_API_KEY environment variable.")
        
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.temperature = temperature
        
        logger.info(f"Initialized Gemini with model: {model}")

    def format_prompt(self, template: str = None, **kwargs) -> str:
        """Format a prompt template with variables"""
        if template is None:
            template = reAct_template
        
        # Extract variable names from template
        variables = re.findall(r'\{(\w+)\}', template)
        
        prompt = PromptTemplate(
            input_variables=variables,
            template=template
        )
        return prompt.format(**kwargs)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text (ported from DeepSeekOllama class)"""
        text = text.strip()
        
        # Try markdown blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Try brace matching if no markdown
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1:
            return text[start:end+1]
            
        return text

    def parse_and_validate(self, raw_output: str, schema: Type[BaseModel]) -> BaseModel:
        """Parse JSON string and validate against Pydantic schema"""
        try:
            cleaned = self._extract_json(raw_output)
            data = json.loads(cleaned)
            validated = schema(**data)
            logger.info(f"Successfully validated output against {schema.__name__}")
            return validated
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Validation failed: {str(e)}")
            raise RuntimeError(f"Output validation failed: {str(e)}") from e

    def _create_example_from_schema(self, schema: Type[BaseModel]) -> str:
        """Create a realistic example JSON from a Pydantic schema."""
        # Simple recursion to create a dummy dictionary based on types
        def _get_example(field_type):
            type_str = str(field_type).lower()
            if 'str' in type_str: return "example_string"
            if 'int' in type_str: return 42
            if 'float' in type_str: return 3.14
            if 'bool' in type_str: return True
            if 'list' in type_str: return ["item1", "item2"]
            if 'dict' in type_str: return {"key": "value"}
            return "value"

        example_data = {}
        for name, field in schema.model_fields.items():
            example_data[name] = _get_example(field.annotation)
            
        return json.dumps(example_data, indent=2)

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        max_retries: int = 2,
        temperature: float = 0.1,
        include_schema_in_prompt: bool = True
    ) -> BaseModel:
        """
        Generate structured output that conforms to a Pydantic schema.
        Uses Gemini's native 'response_mime_type="application/json"' for reliability.
        """
        # 1. Build the Prompt
        enhanced_prompt = prompt
        if include_schema_in_prompt:
            example_json = self._create_example_from_schema(schema)
            enhanced_prompt = f"""{prompt}

CRITICAL: You must return a valid JSON object matching this structure.
Example format:
{example_json}

Return ONLY the JSON.
"""

        full_prompt = f"{self.system_prompt}\n\n{enhanced_prompt}" if self.system_prompt else enhanced_prompt

        # 2. Configure Gemini for JSON Mode
        # We can pass the schema directly to response_schema in the new SDK, 
        # but using standard JSON mode + Pydantic validation is often more flexible.
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            response_mime_type="application/json", 
            response_schema=schema # New feature: Pass Pydantic class directly to Gemini
        )

        last_error = None

        # 3. Retry Loop
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries + 1} to generate structured output")
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=config
                )
                
                # Verify response exists
                if not response.text:
                    raise ValueError("Empty response from Gemini")

                # Parse and Validate
                # Since we used response_schema=schema in config, Gemini usually returns strict JSON.
                # We still run it through our validator to return the Pydantic object.
                return self.parse_and_validate(response.text, schema)

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # If it's a content blocking error, stop retrying
                if "finish_reason" in str(e) and "SAFETY" in str(e):
                    raise RuntimeError("Generation blocked by safety settings") from e

        raise RuntimeError(f"Failed to generate valid structured output after {max_retries + 1} attempts. Last error: {last_error}")

    def call_gemini(self, query: str, template: str = None, use_system_prompt: bool = True) -> str:
        """Call Gemini API with a query (Standard Text)"""
        formatted_query = self.format_prompt(template=template, question=query)
        full_prompt = f"{self.system_prompt}\n\n{formatted_query}" if (use_system_prompt and self.system_prompt) else formatted_query

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_k=64,
                )
            )
            
            thoughts = ""
            answer = ""
            
            # Parsing thinking blocks (structure depends on specific model version)
            # For 2.0 Flash Thinking, it often returns text parts. 
            # Note: The 'thought' attribute in parts is experimental and might vary.
            # This logic assumes the API returns thoughts in a specific way or plain text.
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought') and part.thought:
                     thoughts += f"{part.text}\n" # Sometimes thought text is here
                else:
                    answer += part.text
            
            # If no explicit thought parts were found (common in raw API), 
            # check for <thinking> tags in the text
            if not thoughts and "<thinking>" in answer:
                parts = answer.split("</thinking>")
                if len(parts) > 1:
                    thoughts = parts[0].replace("<thinking>", "").strip()
                    answer = parts[1].strip()

            return thoughts, answer
            
        except Exception as e:
            raise Exception(f"Error calling Gemini API: {str(e)}")

# ==========================================
# Example Usage
# ==========================================

class MovieInfo(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="Release year")
    director: str = Field(description="Name of the director")
    genres: List[str] = Field(description="List of genres")
    is_classic: bool = Field(description="Whether it is considered a classic")

def example_structured():
    print("\n" + "="*80)
    print("STRUCTURED OUTPUT EXAMPLE")
    print("="*80)

    gemini = Gemini(model="gemini-2.0-flash")
    
    prompt = "Tell me about the movie 'Inception'."
    
    try:
        # Generate structured output
        movie_data = gemini.generate_structured(
            prompt=prompt,
            schema=MovieInfo
        )
        
        print(f"Movie: {movie_data.title} ({movie_data.year})")
        print(f"Director: {movie_data.director}")
        print(f"Genres: {', '.join(movie_data.genres)}")
        print(f"Classic: {movie_data.is_classic}")
        print(f"Raw Object: {movie_data}")
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    # example_basic() # From previous code
    example_structured()