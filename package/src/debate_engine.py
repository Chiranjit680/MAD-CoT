# ==================== IMPORTS ====================

import operator
import json
from typing import TypedDict, Annotated, Literal, List, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from gemini import Gemini

from schema import *

# ==================== STATE ====================

class DebateState(TypedDict):
    topic: str
    current_round: int
    max_rounds: int

    debater1_position: str
    debater2_position: str

    debater1_arguments: Annotated[List[dict], operator.add]
    debater2_arguments: Annotated[List[dict], operator.add]

    moderator_notes: Annotated[List[dict], operator.add]
    debate_history: Annotated[List[str], operator.add]

    scores: dict
    winner: str
    final_judgment: Optional[dict]


# ==================== HELPER FUNCTIONS ====================

def print_section(title: str, content: str = ""):
    """Print a formatted section"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)
    if content:
        print(content)


# ==================== MODERATOR NODES ====================

def moderator_opening_node(state: DebateState) -> DebateState:
    """Moderator provides opening statement and rules"""
    
    print_section("MODERATOR OPENING")
    topic= state['topic']
    print(f"Debate Topic: {topic}\n")
    
    llm = Gemini(
       
        temperature=0.2,
        system_prompt="You are a professional debate moderator. Always respond with valid JSON containing the exact fields requested."
    )
    
    prompt = f"""You are moderating a debate on: "{state['topic']}"

Debater 1 will argue: {state['debater1_position']}
Debater 2 will argue: {state['debater2_position']}

The debate will have {state['max_rounds']} rounds.

Provide a JSON response with:
- opening_statement: A welcoming introduction (at least 50 characters)
- rules: An array of exactly 3-5 debate rules (each as a separate string)
- format_description: How the debate will proceed (at least 30 characters)

Example format:
{{
  "opening_statement": "Welcome to today's debate on AI regulation. We have two skilled debaters who will present their positions over three rounds.",
  "rules": [
    "Each debater gets 2 minutes per round",
    "Arguments must be backed by evidence",
    "Remain respectful at all times"
  ],
  "format_description": "The debate will proceed with alternating arguments followed by moderator scoring."
}}"""

    try:
        opening = llm.generate_structured(
            prompt=prompt,
            schema=ModeratorOpeningResponse,
            max_retries=3
        )
        
        state["moderator_notes"].append(opening.model_dump())
        state["debate_history"].append(
            f"🎙️ MODERATOR OPENING:\n{opening.opening_statement}\n\nRULES:\n" + 
            "\n".join(f"  {i+1}. {rule}" for i, rule in enumerate(opening.rules))
        )
        
        print(opening.opening_statement)
        print("\nRULES:")
        for i, rule in enumerate(opening.rules):
            print(f"  {i+1}. {rule}")
        
    except Exception as e:
        print(f"❌ Error in moderator opening: {e}")
        # Fallback
        state["debate_history"].append("🎙️ MODERATOR: Debate starting...")
    
    return state


def moderator_evaluation_node(state: DebateState) -> DebateState:
    """Moderator evaluates the last argument presented"""
    
    # Determine which debater just spoke
    if state["current_round"] % 2 == 1:
        last_debater = "Debater 1"
        last_arg = state["debater1_arguments"][-1] if state["debater1_arguments"] else None
    else:
        last_debater = "Debater 2"
        last_arg = state["debater2_arguments"][-1] if state["debater2_arguments"] else None
    
    if not last_arg:
        return state
    
    print_section(f"MODERATOR EVALUATING {last_debater}")
    
    llm = Gemini(
        
        temperature=0.2,
        system_prompt="You are a professional debate moderator. Always respond with valid JSON containing the exact fields requested."
    )
    
    # Format the argument
    arg_text = last_arg.get('argument', 'No argument provided')
    evidence = last_arg.get('supporting_evidence', [])
    
    prompt = f"""Evaluate {last_debater}'s argument from Round {state['current_round']}.

ARGUMENT:
{arg_text}

EVIDENCE PROVIDED:
{json.dumps(evidence, indent=2)}

Provide a JSON response with:
- round_number: {state['current_round']}
- debater_evaluated: "{last_debater}"
- summary: Brief summary of the argument (at least 30 characters)
- score: Integer from 0 to 10
- validation_notes: Any concerns about factual accuracy (or "No issues found")
- key_points: Array of 2-3 key points made (each as a separate string)

Example format:
{{
  "round_number": {state['current_round']},
  "debater_evaluated": "{last_debater}",
  "summary": "The debater presented a well-structured argument focusing on safety concerns.",
  "score": 7,
  "validation_notes": "All claims appear factually sound.",
  "key_points": [
    "AI safety is paramount",
    "Historical precedents support regulation"
  ]
}}"""

    try:
        evaluation = llm.generate_structured(
            prompt=prompt,
            schema=ModeratorRoundResponse,
            max_retries=3
        )
        
        state["moderator_notes"].append(evaluation.model_dump())
        
        # Update scores
        if last_debater == "Debater 1":
            state["scores"]["debater1"].append(evaluation.score)
        else:
            state["scores"]["debater2"].append(evaluation.score)
        
        # Add to history
        history_entry = f"⚖️ MODERATOR EVALUATION - Round {state['current_round']}\n"
        history_entry += f"Evaluating: {last_debater}\n"
        history_entry += f"Summary: {evaluation.summary}\n"
        history_entry += f"Score: {evaluation.score}/10\n"
        history_entry += f"Key Points: {', '.join(evaluation.key_points)}"
        
        state["debate_history"].append(history_entry)
        
        print(f"Summary: {evaluation.summary}")
        print(f"Score: {evaluation.score}/10")
        print(f"Key Points: {', '.join(evaluation.key_points)}")
        
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        # Fallback scoring
        default_score = 5
        if last_debater == "Debater 1":
            state["scores"]["debater1"].append(default_score)
        else:
            state["scores"]["debater2"].append(default_score)
    
    return state


def moderator_final_node(state: DebateState) -> DebateState:
    """Moderator provides final judgment and declares winner"""
    
    print_section("FINAL JUDGMENT")
    
    llm = Gemini(
        
        temperature=0.2,
        system_prompt="You are a professional debate moderator. Always respond with valid JSON containing the exact fields requested."
    )
    
    d1_total = sum(state["scores"]["debater1"])
    d2_total = sum(state["scores"]["debater2"])
    
    # Format arguments for display
    d1_args = "\n".join([
        f"Round {i+1}: {arg.get('argument', 'N/A')}"
        for i, arg in enumerate(state["debater1_arguments"])
    ])
    d2_args = "\n".join([
        f"Round {i+1}: {arg.get('argument', 'N/A')}"
        for i, arg in enumerate(state["debater2_arguments"])
    ])
    
    prompt = f"""Provide final judgment for the debate on "{state['topic']}".

DEBATER 1 ({state['debater1_position']}):
Total Score: {d1_total}
Arguments:
{d1_args}

DEBATER 2 ({state['debater2_position']}):
Total Score: {d2_total}
Arguments:
{d2_args}

Provide a JSON response with:
- final_summary: Comprehensive summary of the debate (at least 100 characters)
- winner: Must be exactly "Debater 1", "Debater 2", or "Tie"
- reasoning: Explanation for your decision (at least 50 characters)
- debater1_total_score: {d1_total}
- debater2_total_score: {d2_total}
- key_moments: Array of 2-3 key moments (each as a separate string)

Example format:
{{
  "final_summary": "This debate covered important aspects of AI regulation with both sides presenting compelling arguments backed by evidence.",
  "winner": "Debater 1",
  "reasoning": "Debater 1 provided more concrete evidence and addressed counterarguments more effectively.",
  "debater1_total_score": {d1_total},
  "debater2_total_score": {d2_total},
  "key_moments": [
    "Debater 1's safety argument in round 2",
    "Debater 2's innovation point in round 1"
  ]
}}"""

    try:
        final = llm.generate_structured(
            prompt=prompt,
            schema=ModeratorFinalResponse,
            max_retries=3
        )
        
        state["final_judgment"] = final.model_dump()
        state["winner"] = final.winner
        
        # Display
        print(f"\n{final.final_summary}\n")
        print(f"SCORES:")
        print(f"  Debater 1: {final.debater1_total_score}")
        print(f"  Debater 2: {final.debater2_total_score}")
        print(f"\nWINNER: {final.winner}")
        print(f"\nREASONING:\n{final.reasoning}")
        
        state["debate_history"].append(
            f"🏆 FINAL JUDGMENT\n\n{final.final_summary}\n\nWINNER: {final.winner}\n\n{final.reasoning}"
        )
        
    except Exception as e:
        print(f"❌ Error in final judgment: {e}")
        # Determine winner by score
        if d1_total > d2_total:
            winner = "Debater 1"
        elif d2_total > d1_total:
            winner = "Debater 2"
        else:
            winner = "Tie"
        
        state["winner"] = winner
        print(f"WINNER (by score): {winner}")
    
    return state


# ==================== DEBATER NODES ====================

def debater1_node(state: DebateState) -> DebateState:
    """Debater 1 presents their argument"""
    
    print_section(f"DEBATER 1 - Round {state['current_round']}")
    
    llm = Gemini(
       
        temperature=0.7,
        system_prompt=f"You are Debater 1. You argue that: {state['debater1_position']}. Always respond with valid JSON."
    )
    
    # Get opponent's last argument if available
    opponent_last = ""
    if state["debater2_arguments"]:
        last = state["debater2_arguments"][-1]
        opponent_last = f"\n\nYour opponent's last argument:\n{last.get('argument', 'N/A')}"
    
    prompt = f"""You are Debater 1 in Round {state['current_round']} of {state['max_rounds']}.

Debate topic: "{state['topic']}"
Your position: {state['debater1_position']}
{opponent_last}

Provide a JSON response with:
- argument: Your main argument (at least 50 characters)
- supporting_evidence: Array of 2-4 evidence points (each as a separate string)
- counterpoints: Array of counterpoints to opponent (optional)
- conclusion: Brief conclusion (optional)

Example format:
{{
  "argument": "Government regulation of AI is essential because uncontrolled development poses significant risks to privacy and safety.",
  "supporting_evidence": [
    "Historical precedent: Financial sector regulation prevented market crashes",
    "Current AI systems lack transparency in decision-making",
    "EU's AI Act shows feasibility of effective regulation"
  ],
  "counterpoints": [],
  "conclusion": "Therefore, measured regulation is necessary."
}}"""

    try:
        arg = llm.generate_structured(
            prompt=prompt,
            schema=DebaterResponse,
            max_retries=3
        )
        
        state["debater1_arguments"].append(arg.model_dump())
        
        print(f"\nARGUMENT:\n{arg.argument}\n")
        print("EVIDENCE:")
        for i, ev in enumerate(arg.supporting_evidence):
            print(f"  {i+1}. {ev}")
        
        state["debate_history"].append(
            f"💬 DEBATER 1 - Round {state['current_round']}\n\n{arg.argument}\n\nEvidence:\n" +
            "\n".join(f"  • {e}" for e in arg.supporting_evidence)
        )
        
    except Exception as e:
        print(f"❌ Error generating argument: {e}")
        # Fallback
        fallback = {
            "argument": f"I support {state['debater1_position']} for important reasons.",
            "supporting_evidence": ["Evidence point 1", "Evidence point 2"],
            "counterpoints": [],
            "conclusion": ""
        }
        state["debater1_arguments"].append(fallback)
    
    return state


def debater2_node(state: DebateState) -> DebateState:
    """Debater 2 presents their argument"""
    
    print_section(f"DEBATER 2 - Round {state['current_round']}")
    
    llm = Gemini(
       
        temperature=0.7,
        system_prompt=f"You are Debater 2. You argue that: {state['debater2_position']}. Always respond with valid JSON."
    )
    
    # Get opponent's last argument
    opponent_last = ""
    if state["debater1_arguments"]:
        last = state["debater1_arguments"][-1]
        opponent_last = f"\n\nYour opponent's last argument:\n{last.get('argument', 'N/A')}"
    
    prompt = f"""You are Debater 2 in Round {state['current_round']} of {state['max_rounds']}.

Debate topic: "{state['topic']}"
Your position: {state['debater2_position']}
{opponent_last}

Provide a JSON response with:
- argument: Your main argument (at least 50 characters)
- supporting_evidence: Array of 2-4 evidence points (each as a separate string)
- counterpoints: Array of counterpoints to opponent (optional)
- conclusion: Brief conclusion (optional)

Example format:
{{
  "argument": "AI regulation by governments would stifle innovation and slow technological progress that benefits society.",
  "supporting_evidence": [
    "Tech industry self-regulation has proven effective",
    "Regulatory frameworks cannot keep pace with AI advancement",
    "Over-regulation drove innovation offshore in other industries"
  ],
  "counterpoints": ["Regulation doesn't prevent innovation if done correctly"],
  "conclusion": "Free market principles should guide AI development."
}}"""

    try:
        arg = llm.generate_structured(
            prompt=prompt,
            schema=DebaterResponse,
            max_retries=3
        )
        
        state["debater2_arguments"].append(arg.model_dump())
        
        print(f"\nARGUMENT:\n{arg.argument}\n")
        print("EVIDENCE:")
        for i, ev in enumerate(arg.supporting_evidence):
            print(f"  {i+1}. {ev}")
        
        state["debate_history"].append(
            f"💬 DEBATER 2 - Round {state['current_round']}\n\n{arg.argument}\n\nEvidence:\n" +
            "\n".join(f"  • {e}" for e in arg.supporting_evidence)
        )
        
    except Exception as e:
        print(f"❌ Error generating argument: {e}")
        # Fallback
        fallback = {
            "argument": f"I support {state['debater2_position']} for important reasons.",
            "supporting_evidence": ["Evidence point 1", "Evidence point 2"],
            "counterpoints": [],
            "conclusion": ""
        }
        state["debater2_arguments"].append(fallback)
    
    return state


# ==================== ROUTING ====================

def increment_round(state: DebateState) -> DebateState:
    """Increment the round counter"""
    state["current_round"] += 1
    return state


def route_after_opening(state: DebateState) -> Literal["increment_round"]:
    """After opening, always increment to round 1"""
    return "increment_round"


def route_after_increment(state: DebateState) -> Literal["debater1", "debater2", "moderator_final"]:
    """Decide which debater speaks next or if debate is over"""
    if state["current_round"] > state["max_rounds"]:
        return "moderator_final"
    
    # Odd rounds: Debater 1, Even rounds: Debater 2
    if state["current_round"] % 2 == 1:
        return "debater1"
    else:
        return "debater2"


def route_after_evaluation(state: DebateState) -> Literal["increment_round", "moderator_final"]:
    """After evaluation, either increment round or go to final judgment"""
    if state["current_round"] >= state["max_rounds"]:
        return "moderator_final"
    return "increment_round"


# ==================== GRAPH ====================

def build_graph():
    """Build the debate workflow graph with separate moderator nodes"""
    g = StateGraph(DebateState)

    # Add nodes
    g.add_node("moderator_opening", moderator_opening_node)
    g.add_node("moderator_evaluation", moderator_evaluation_node)
    g.add_node("moderator_final", moderator_final_node)
    g.add_node("debater1", debater1_node)
    g.add_node("debater2", debater2_node)
    g.add_node("increment_round", increment_round)

    # Flow: START -> opening -> increment -> debater -> evaluation -> increment -> ...
    
    # Start with moderator opening
    g.add_edge(START, "moderator_opening")
    
    # After opening, increment to round 1
    g.add_edge("moderator_opening", "increment_round")
    
    # After increment, route to appropriate debater or final judgment
    g.add_conditional_edges(
        "increment_round",
        route_after_increment,
        {
            "debater1": "debater1",
            "debater2": "debater2",
            "moderator_final": "moderator_final"
        }
    )
    
    # After debater speaks, go to evaluation
    g.add_edge("debater1", "moderator_evaluation")
    g.add_edge("debater2", "moderator_evaluation")
    
    # After evaluation, check if we need another round or final judgment
    g.add_conditional_edges(
        "moderator_evaluation",
        route_after_evaluation,
        {
            "increment_round": "increment_round",
            "moderator_final": "moderator_final"
        }
    )
    
    # Final judgment ends the debate
    g.add_edge("moderator_final", END)

    return g.compile(checkpointer=MemorySaver())


# ==================== MAIN ====================

if __name__ == "__main__":
    # Initialize debate state
    initial_state: DebateState = {
        "topic": "AU will replace human programmers?",
        "current_round": 0,
        "max_rounds": 3,
        "debater1_position": "Yes, regulation is necessary",
        "debater2_position": "No, regulation will slow innovation",
        "debater1_arguments": [],
        "debater2_arguments": [],
        "moderator_notes": [],
        "debate_history": [],
        "scores": {"debater1": [], "debater2": []},
        "winner": "",
        "final_judgment": None,
    }

    # Build and run
    print_section("AI DEBATE SYSTEM", f"Topic: {initial_state['topic']}")
    print(f"Rounds: {initial_state['max_rounds']}")
    
    app = build_graph()
    
    try:
        final_state = app.invoke(
            initial_state,
            {"configurable": {"thread_id": "debate_1"}}
        )

        print_section("DEBATE COMPLETE")
        print(f"Winner: {final_state.get('winner', 'Not determined')}")
        print(f"Final scores - D1: {sum(final_state['scores']['debater1'])}, D2: {sum(final_state['scores']['debater2'])}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Debate interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during debate: {e}")
        import traceback
        traceback.print_exc()