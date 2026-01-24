import json
import re
import logging
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field, ValidationError
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekOllama:
    """
    A wrapper for DeepSeek models via Ollama with built-in JSON parsing
    and Pydantic schema validation.
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = 120
    ):
        """
        Initialize the DeepSeek Ollama wrapper.
        
        Args:
            model: The Ollama model name
            temperature: Sampling temperature (0.0 = deterministic)
            system_prompt: Default system prompt for all requests
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize LLM
        llm_kwargs = {
            "model": model,
            "temperature": temperature,
        }
        
        if max_tokens:
            llm_kwargs["num_predict"] = max_tokens
        
        if timeout:
            llm_kwargs["timeout"] = timeout
            
        self.llm = ChatOllama(**llm_kwargs)
        
        self.system_prompt = system_prompt or (
            "You are a precise reasoning assistant. "
            "Provide clear, structured, and accurate responses."
        )
        
        logger.info(f"Initialized DeepSeekOllama with model: {model}")
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text that may contain markdown code blocks or extra text.
        
        Args:
            text: Raw text that may contain JSON
            
        Returns:
            Cleaned JSON string
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Try to find JSON in markdown code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                logger.debug("Extracted JSON from markdown code block")
                # Try to parse it to verify
                try:
                    json.loads(extracted)
                    return extracted
                except:
                    continue
        
        # Try to find JSON object or array by looking for balanced braces/brackets
        # Find the first { or [
        start_brace = text.find('{')
        start_bracket = text.find('[')
        
        if start_brace == -1 and start_bracket == -1:
            return text  # No JSON found, return original
        
        # Determine which comes first
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            # Object starts first
            start = start_brace
            open_char = '{'
            close_char = '}'
        else:
            # Array starts first
            start = start_bracket
            open_char = '['
            close_char = ']'
        
        # Find matching closing brace/bracket
        depth = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        # Found matching closing character
                        extracted = text[start:i+1]
                        logger.debug("Extracted JSON using brace matching")
                        return extracted
        
        # If we get here, no complete JSON was found
        logger.debug("No complete JSON found, returning original text")
        return text
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Override default system prompt
            temperature: Override default temperature
            
        Returns:
            Generated text response
        """
        try:
            # Use provided system prompt or fall back to default
            sys_prompt = system_prompt or self.system_prompt
            
            messages = [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Override temperature if provided
            if temperature is not None:
                original_temp = self.llm.temperature
                self.llm.temperature = temperature
                response = self.llm.invoke(messages)
                self.llm.temperature = original_temp
            else:
                response = self.llm.invoke(messages)
            
            logger.info(f"Generated response (length: {len(response.content)} chars)")
            return response.content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e
    
    def parse_and_validate(
        self,
        raw_output: str,
        schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Parse JSON string and validate against Pydantic schema.
        
        Args:
            raw_output: Raw text output from LLM
            schema: Pydantic model class to validate against
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            RuntimeError: If JSON parsing or validation fails
        """
        try:
            # Extract and clean JSON
            cleaned = self._extract_json(raw_output)
            
            # Parse JSON
            data = json.loads(cleaned)
            
            # Validate with Pydantic schema
            validated = schema(**data)
            
            logger.info(f"Successfully validated output against {schema.__name__}")
            return validated
            
        except json.JSONDecodeError as e:
            error_msg = (
                f"Output is not valid JSON.\n\n"
                f"Error: {str(e)}\n\n"
                f"Cleaned output:\n{cleaned[:500]}...\n\n"
                f"Raw output:\n{raw_output[:500]}..."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
        except ValidationError as e:
            error_msg = (
                f"Output violates schema {schema.__name__}.\n\n"
                f"Validation errors:\n{str(e)}\n\n"
                f"Parsed data: {json.dumps(data, indent=2)[:500]}...\n\n"
                f"Raw output:\n{raw_output[:500]}..."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _create_example_from_schema(self, schema: Type[BaseModel]) -> str:
        """
        Create a realistic example JSON from a Pydantic schema.
        
        Args:
            schema: Pydantic model class
            
        Returns:
            Example JSON string
        """
        example_data = {}
        
        for field_name, field_info in schema.model_fields.items():
            field_type = field_info.annotation
            description = field_info.description or ""
            
            # Generate example based on type
            if field_type == str or 'str' in str(field_type):
                example_data[field_name] = f"Example {field_name.replace('_', ' ')}"
            elif field_type == int or 'int' in str(field_type):
                example_data[field_name] = 5
            elif field_type == float or 'float' in str(field_type):
                example_data[field_name] = 0.5
            elif field_type == bool or 'bool' in str(field_type):
                example_data[field_name] = True
            elif 'list' in str(field_type).lower() or 'List' in str(field_type):
                example_data[field_name] = ["item1", "item2", "item3"]
            elif 'dict' in str(field_type).lower() or 'Dict' in str(field_type):
                example_data[field_name] = {"key": "value"}
            else:
                example_data[field_name] = f"<{field_name}>"
        
        return json.dumps(example_data, indent=2)
    
    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        include_schema_in_prompt: bool = True
    ) -> BaseModel:
        """
        Generate structured output that conforms to a Pydantic schema.
        
        Args:
            prompt: User prompt
            schema: Pydantic model class defining expected output structure
            max_retries: Number of retry attempts on validation failure
            system_prompt: Override default system prompt
            temperature: Override default temperature
            include_schema_in_prompt: Whether to include schema in the prompt
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Create example from schema
        example_json = self._create_example_from_schema(schema)
        
        # Build enhanced prompt with clear instructions
        if include_schema_in_prompt:
            # Get field descriptions
            field_descriptions = []
            for field_name, field_info in schema.model_fields.items():
                desc = field_info.description or field_name
                field_type = str(field_info.annotation).replace('typing.', '')
                required = "required" if field_info.is_required() else "optional"
                field_descriptions.append(f"  - {field_name} ({field_type}, {required}): {desc}")
            
            enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
You MUST respond with ONLY a valid JSON object. Do not include any explanatory text, markdown formatting, or comments.

Required JSON fields:
{chr(10).join(field_descriptions)}

Example response format (use this exact structure):
{example_json}

Remember:
1. Start your response with {{ (opening brace)
2. End your response with }} (closing brace)
3. Include ALL required fields
4. Use proper JSON syntax (quoted strings, correct types)
5. Do NOT wrap the JSON in markdown code blocks
6. Do NOT include any text before or after the JSON"""
        else:
            enhanced_prompt = prompt
        
        # Override system prompt for structured generation
        structured_system_prompt = system_prompt or self.system_prompt
        structured_system_prompt += " You must respond with valid JSON only, with no additional text or formatting."
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries + 1} to generate structured output")
                
                # Generate response
                raw_output = self.generate(
                    enhanced_prompt,
                    system_prompt=structured_system_prompt,
                    temperature=temperature
                )
                
                logger.debug(f"Raw output preview: {raw_output[:200]}...")
                
                # Parse and validate
                result = self.parse_and_validate(raw_output, schema)
                
                logger.info(f"Successfully generated structured output on attempt {attempt + 1}")
                return result
                
            except RuntimeError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)[:200]}")
                
                if attempt < max_retries:
                    # Add specific feedback for retry
                    enhanced_prompt = f"""{prompt}

RETRY ATTEMPT {attempt + 2}/{max_retries + 1}

Previous attempt failed. Common mistakes to avoid:
1. Don't return the schema definition itself
2. Don't include explanatory text
3. Don't use markdown code blocks
4. Start with {{ and end with }}
5. Fill in actual values, not field names

Required response format (fill with actual content):
{example_json}

Your response must be ONLY the JSON object with actual values."""
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        # All retries exhausted
        raise RuntimeError(
            f"Failed to generate valid structured output after {max_retries + 1} attempts. "
            f"Last error: {str(last_error)}"
        )
    
    def generate_with_reasoning(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate output with explicit reasoning chain (useful for DeepSeek R1 models).
        
        Args:
            prompt: User prompt
            schema: Optional Pydantic schema for structured output
            system_prompt: Override default system prompt
            
        Returns:
            Dictionary with 'reasoning' and 'output' keys
        """
        reasoning_prompt = f"""{prompt}

Please provide your response in the following format:
1. First, show your reasoning and thought process
2. Then, provide your final answer

Structure your response clearly."""
        
        raw_output = self.generate(
            reasoning_prompt,
            system_prompt=system_prompt
        )
        
        # Try to split reasoning and answer
        parts = raw_output.split("\n\n", 1)
        
        if len(parts) == 2:
            reasoning, output = parts
        else:
            reasoning = ""
            output = raw_output
        
        result = {
            "reasoning": reasoning.strip(),
            "output": output.strip()
        }
        
        # If schema provided, validate the output
        if schema:
            result["structured_output"] = self.parse_and_validate(output, schema)
        
        return result
    
    def batch_generate_structured(
        self,
        prompts: list[str],
        schema: Type[BaseModel],
        max_retries: int = 2
    ) -> list[BaseModel]:
        """
        Generate structured outputs for multiple prompts.
        
        Args:
            prompts: List of prompts
            schema: Pydantic model class for validation
            max_retries: Retries per prompt
            
        Returns:
            List of validated Pydantic model instances
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing batch item {i + 1}/{len(prompts)}")
            try:
                result = self.generate_structured(
                    prompt=prompt,
                    schema=schema,
                    max_retries=max_retries
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process batch item {i + 1}: {str(e)}")
                results.append(None)
        
        return results


# Example usage and test cases
if __name__ == "__main__":
    # Define example schemas
    class Reasoning(BaseModel):
        """Schema for reasoning output"""
        steps: list[str] = Field(description="List of reasoning steps")
        conclusion: str = Field(description="Final conclusion")
        confidence: float = Field(ge=0, le=1, description="Confidence score between 0 and 1")
    
    # Initialize the wrapper
    llm = DeepSeekOllama(
        model="deepseek-r1:1.5b",
        temperature=0.0
    )
    
    # Test structured output
    print("=== Testing Structured Output ===")
    result = llm.generate_structured(
        prompt="Explain why the sky is blue in simple terms",
        schema=Reasoning
    )
    print(f"Conclusion: {result.conclusion}")
    print(f"Confidence: {result.confidence}")
    print(f"Steps: {result.steps}")