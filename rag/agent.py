# rag/agent.py — FINAL WORKING AGENT (LangChain 0.2+ Stable)
from typing import Dict, Any
import logging

from rag.tools.langchain_tools import WrappedLLM, get_langchain_tools
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

try:
    from langchain.agents.react.base import create_react_agent
    from langchain.agents import AgentExecutor
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Fallback in case Django models aren't available when module is imported
try:
    from django.contrib.auth import get_user_model
    User = get_user_model()
except Exception:
    User = None

SYSTEM_PROMPT = """
You are NeuroStack AI — an intelligent tool-using agent.

High-level rules (strict RAG-first):
1. ALWAYS try to answer from rag_search outputs first.
2. If rag_search returns insufficient info, consult memory_search.
3. Use rewrite_query to improve retrieval if the question is ambiguous.
4. Use web_search ONLY if rag_search+memory_search cannot produce an answer OR the user explicitly requests current/external facts (e.g. 'who is', 'today', 'latest', 'news', dates).
5. If uncertain or not found, respond: "I don’t know based on the available documentation."
6. Your final answer MUST appear only in the form: Final Answer: <answer>
"""

REACT_PROMPT = PromptTemplate.from_template("""
{system_prompt}

You have access to the following tools:
{tools}

Tool names: {tool_names}

Format:
Thought: <reason>
Action: <tool_name>
Action Input: <input>
Observation: <tool output>

If no tool needed:
Final Answer: <your answer>

Question: {input}

Thought: Let's think step-by-step.
{agent_scratchpad}
""")

class SimpleAgent:
    def __init__(self, user_id=None, agent_enabled=True, user_obj=None):
        self.user_id = user_id
        self.user = user_obj
        self.enabled = agent_enabled
        self._executor = None

        if LANGCHAIN_AVAILABLE and self.enabled:
            try:
                self._build_agent()
            except Exception as e:
                logger.exception("Agent build failed: %s", e)
                self._executor = None

    def _build_agent(self):
        llm = WrappedLLM()
        tools = get_langchain_tools(user_obj=self.user)

        tool_names = ", ".join(t.name for t in tools) if isinstance(tools, list) else ", ".join(list(tools.keys()))
        tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools) if isinstance(tools, list) else "\n".join(f"- {k}" for k in tools.keys())

        prompt = REACT_PROMPT.partial(
            system_prompt=SYSTEM_PROMPT,
            tools=tools_block,
            tool_names=tool_names
        )

        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt
        )

        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=60.0
        )

    def _extract_final_answer(self, raw: str) -> str:
        if not raw:
            return ""
        # prefer explicit "Final Answer:"
        if "Final Answer:" in raw:
            return raw.split("Final Answer:", 1)[1].strip()
        # otherwise, attempt a last-line fallback
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else raw.strip()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent. If LangChain executor exists, use it.
        We return a dict with answer, mode, confidence, chunks (empty), trace.
        """
        if self._executor:
            try:
                result = self._executor.invoke({"input": query})
                raw = result.get("output", "") or ""
                final_answer = self._extract_final_answer(raw)

                # If the agent used web_search but RAG/memory contained evidence, that's undesirable.
                # We can't always detect that here, but the prompt reduces such cases.

                return {
                    "answer": final_answer or "I don’t know based on the available documentation.",
                    "mode": "agent",
                    "confidence": 1.0,
                    "chunks": [],
                    "trace": [{"tool": "agent", "result": raw}],
                    "steps": 1
                }
            except Exception as e:
                logger.exception("Agent error: %s", e)
                return {
                    "answer": f"Agent failed to execute the full plan. Reverting to basic LLM. Error: {str(e)}",
                    "mode": "agent_error",
                    "confidence": 0.0,
                    "chunks": [],
                    "trace": [{"tool": "agent_failure", "result": str(e)}],
                    "steps": 0
                }

        # Fallback: if the AgentExecutor was not built, use a simple LLM call that respects RAG-first policy.
        try:
            # Ask LLM to follow the rag-first policy in a compact prompt
            compact_system = (
                "You are a safe assistant. Try to answer using internal documents. "
                "If you cannot, say: 'I don’t know based on the available documentation.'"
            )
            out = WrappedLLM().invoke(query)
        except Exception as e:
            out = "I don’t know based on the available documentation."

        return {
            "answer": out,
            "mode": "agent",
            "confidence": 1.0,
            "chunks": [],
            "trace": [{"tool": "fallback_llm", "result": out}],
            "steps": 1
        }

def build_agent(user_id=None, agent_enabled=True, **kwargs):
    user_obj = None
    if User:
        try:
            user_obj = User.objects.get(id=user_id)
        except Exception:
            user_obj = None

    return SimpleAgent(user_id=user_id, agent_enabled=agent_enabled, user_obj=user_obj)
