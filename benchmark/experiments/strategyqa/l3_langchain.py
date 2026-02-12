"""
L3 LangChain ReAct Agent for StrategyQA.

Uses LangChain's agent with a broad set of tools, letting the LLM
freely choose which tools to use. The goal is to study tool selection
behavior rather than prescribing a fixed workflow.

Available tools (all provided by default):
- DuckDuckGo web search
- Wikipedia lookup
- StrategyQA knowledge base search (TF-IDF RAG)
- Python calculator (math/computation)

Key differences from hand-rolled L3:
- LangChain manages the tool-calling loop and error recovery
- Agent has access to real external tools (web search, Wikipedia)
- Tool selection is fully agent-driven — we analyze choices post-hoc
- Full message trace is captured for analysis
"""

import os
import time
from typing import Optional, Dict, Any, List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .base import StrategyQAExperiment, StrategyQAResult
from ...dataset.strategyqa_loader import StrategyQAInstance
from ...metrics.cost import calculate_cost
from ptool_framework.llm_backend import LLMConfig


# ============================================================================
# Provider base URLs
# ============================================================================

PROVIDER_BASE_URLS = {
    "together": "https://api.together.xyz/v1",
    "openai": None,  # Default OpenAI endpoint
    "groq": "https://api.groq.com/openai/v1",
}


# ============================================================================
# Tool definitions
# ============================================================================

def _make_rag_tool():
    """Create the StrategyQA knowledge base search tool."""
    from .trace_builder import get_paragraph_retriever

    @tool
    def strategyqa_knowledge_search(query: str) -> str:
        """Search the StrategyQA knowledge base of Wikipedia paragraphs for relevant facts.
        Use this when you need specific factual information that may be in the knowledge base.
        Input should be a search query about the topic you need facts on."""
        retriever = get_paragraph_retriever()
        retriever.load()
        results = retriever.retrieve(query, top_k=3)
        if not results:
            return "No relevant paragraphs found."
        return "\n\n".join(f"[{r['title']}]: {r['content']}" for r in results)

    return strategyqa_knowledge_search


def _make_search_tool():
    """Create the DuckDuckGo web search tool."""
    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun()


def _make_wikipedia_tool():
    """Create the Wikipedia lookup tool."""
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
    return WikipediaQueryRun(api_wrapper=api_wrapper)


def _make_python_tool():
    """Create the Python calculator tool for math/computation."""

    @tool
    def python_calculator(expression: str) -> str:
        """Evaluate a Python math expression. Use this for any numerical computation.
        Input should be a valid Python expression like '2**10' or '299792458 / 1000'."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return python_calculator


def get_default_tools() -> List:
    """Get the default set of tools for the LangChain agent."""
    return [
        _make_search_tool(),
        _make_wikipedia_tool(),
        _make_rag_tool(),
        _make_python_tool(),
    ]


# ============================================================================
# L3 LangChain Experiment
# ============================================================================

SYSTEM_PROMPT_OPTIONAL = (
    "You are a reasoning agent that answers yes/no questions. "
    "Use the available tools to look up facts, search for information, "
    "and perform calculations as needed. "
    "Once you have enough information, respond with ONLY 'Yes' or 'No'."
)

SYSTEM_PROMPT_MANDATORY = (
    "You are a research agent that answers yes/no questions. "
    "You MUST call at least one tool before answering. "
    "NEVER answer directly from your own knowledge. "
    "Your process: 1) Identify what facts you need. "
    "2) Use tools (search, wikipedia, knowledge base, calculator) to find those facts. "
    "3) Only after receiving tool results, give your final 'Yes' or 'No' answer. "
    "If you answer without calling a tool first, your answer will be marked INVALID."
)


class L3_LangChain(StrategyQAExperiment):
    """
    L3-LangChain: Agent with open tool access via LangChain.

    Gives the LLM a broad toolkit and lets it freely choose which tools
    to use. The full message trace (including tool calls, reasoning, and
    observations) is captured for post-hoc analysis of tool selection.

    Args:
        require_tool_use: If True, system prompt mandates at least one tool
            call before answering. If False, agent may answer directly.
    """

    def __init__(
        self,
        model: str = "deepseek-v3-0324",
        max_iterations: int = 10,
        tools: Optional[List] = None,
        require_tool_use: bool = False,
    ):
        super().__init__(model)
        self.max_iterations = max_iterations
        self.require_tool_use = require_tool_use

        # Resolve model config from LLMS.json
        config = LLMConfig.load()
        model_config = config.models.get(model)

        if model_config is None:
            raise ValueError(
                f"Model '{model}' not found in LLMS.json. "
                f"Available: {list(config.models.keys())}"
            )

        base_url = PROVIDER_BASE_URLS.get(model_config.provider)
        api_key = model_config.get_api_key()

        if not api_key:
            raise ValueError(
                f"API key not found for model '{model}'. "
                f"Set {model_config.api_key_env} in your environment."
            )

        # Create LangChain LLM
        self.llm = ChatOpenAI(
            model=model_config.model_id,
            base_url=base_url,
            api_key=api_key,
            temperature=0,
        )

        # Tools — use provided list or defaults
        self.tools = tools if tools is not None else get_default_tools()

        # Select system prompt based on require_tool_use
        system_prompt = SYSTEM_PROMPT_MANDATORY if require_tool_use else SYSTEM_PROMPT_OPTIONAL

        # Create agent (LangGraph-based, LangChain 1.x)
        self.agent = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

        # Track tool usage across all instances
        self.tool_usage_stats: Dict[str, int] = {}

    def run_instance(self, instance: StrategyQAInstance) -> StrategyQAResult:
        """Run LangChain agent on a single instance."""
        start_time = time.time()

        question = f"Answer this yes/no question: {instance.question}"

        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": self.max_iterations * 2},
            )
            latency_ms = (time.time() - start_time) * 1000

            messages = result["messages"]

            # Extract final answer from last AI message
            final_answer_text = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    final_answer_text = msg.content
                    break

            predicted = self._extract_yes_no(final_answer_text)

            # Build trace from message history
            trace = self._build_trace(messages)

            # Update tool usage stats
            for tool_name in trace["tools_used"]:
                self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + 1

            # Estimate tokens from message content
            input_tokens = 0
            output_tokens = 0
            for msg in messages:
                content_len = len(msg.content) // 4 if msg.content else 0
                if isinstance(msg, (HumanMessage, ToolMessage)):
                    input_tokens += content_len
                elif isinstance(msg, AIMessage):
                    output_tokens += content_len

            cost = calculate_cost(input_tokens, output_tokens, self.model)

            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=predicted,
                ground_truth=instance.answer,
                is_correct=predicted == instance.answer if predicted is not None else False,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost.cost_usd,
                num_steps=trace["num_tool_calls"],
                raw_response=final_answer_text,
                trace=trace,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return StrategyQAResult(
                qid=instance.qid,
                question=instance.question,
                predicted_answer=None,
                ground_truth=instance.answer,
                is_correct=False,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _build_trace(self, messages: list) -> Dict[str, Any]:
        """Build trace dict from LangChain message history.

        Captures two views:
        - 'steps': structured tool-call / response steps (for analysis)
        - 'full_conversation': ordered list of every message for debugging
        """
        steps = []
        tools_used = []
        full_conversation = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                full_conversation.append({
                    "role": "human",
                    "content": msg.content,
                })

            elif isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "type": "tool_call",
                        "tool": tc["name"],
                        "tool_input": tc["args"],
                        "reasoning": msg.content or "",
                    })
                    tools_used.append(tc["name"])
                full_conversation.append({
                    "role": "ai",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in msg.tool_calls
                    ],
                })

            elif isinstance(msg, ToolMessage):
                tool_output = str(msg.content)
                # Attach observation to the last step
                if steps and steps[-1]["tool"] == msg.name:
                    steps[-1]["observation"] = tool_output[:500]
                full_conversation.append({
                    "role": "tool",
                    "tool_name": msg.name,
                    "content": tool_output,
                })

            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                steps.append({
                    "type": "response",
                    "content": msg.content,
                })
                full_conversation.append({
                    "role": "ai",
                    "content": msg.content,
                })

        return {
            "steps": steps,
            "tools_used": tools_used,
            "num_tool_calls": len(tools_used),
            "full_conversation": full_conversation,
        }

    def _extract_yes_no(self, text: str) -> Optional[bool]:
        """Extract Yes/No from agent output."""
        if not text:
            return None
        text_lower = text.strip().lower()
        if text_lower in ("yes", "yes."):
            return True
        if text_lower in ("no", "no."):
            return False
        return self.extract_boolean(text)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary with tool usage analysis."""
        summary = super().get_summary()
        summary["tool_usage"] = dict(sorted(
            self.tool_usage_stats.items(),
            key=lambda x: x[1],
            reverse=True,
        ))
        summary["tools_available"] = [t.name for t in self.tools]
        summary["require_tool_use"] = self.require_tool_use
        return summary


class L3_LangChainMandatory(L3_LangChain):
    """L3-LangChain with mandatory tool use. Registered as 'L3-langchain-tools'."""

    def __init__(self, model: str = "deepseek-v3-0324", max_iterations: int = 10):
        super().__init__(model, max_iterations=max_iterations, require_tool_use=True)
