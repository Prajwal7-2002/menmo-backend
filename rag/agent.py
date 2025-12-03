# rag/agent.py
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

# Agent prompt: allow agent discretion — agent decides when web is necessary.
SYSTEM_PROMPT = """
You are NeuroStack AI — an intelligent tool-using agent.

GOAL:
- Prefer internal evidence (RAG) and memory for answers.
- You MAY use web_search when internal evidence and memory are insufficient or
  when the user explicitly requests up-to-date world knowledge.
- If unsure and no evidence, prefer a conservative response: "I don’t know based on the available documentation."

FORMAT RULES:
- When using tools follow this format:
  Thought: <reason>
  Action: <tool_name>
  Action Input: <input>
  Observation: <tool output>

- If no tool is needed produce:
  Final Answer: <your answer>
"""

REACT_PROMPT = PromptTemplate.from_template("""
{system_prompt}

You have access to the following tools:
{tools}

Tool names: {tool_names}

Use this format:
Thought: <reason>
Action: <tool_name>
Action Input: <input>
Observation: <tool output>

If no tool is needed:
Final Answer: <your answer>

Question: {input}

Thought: Let's think step-by-step.
{agent_scratchpad}
""")

class SimpleAgent:
    def __init__(self, user_id=None, agent_enabled=True, user_obj=None, allow_web: bool = True):
        self.user_id = user_id
        self.user = user_obj
        self.enabled = agent_enabled
        self.allow_web = allow_web
        self._executor = None

        if LANGCHAIN_AVAILABLE and self.enabled:
            try:
                self._build_agent()
            except Exception as e:
                logger.exception("Agent build failed: %s", e)
                self._executor = None

    def _build_agent(self):
        llm = WrappedLLM()
        tools = get_langchain_tools(user_obj=self.user, allow_web=self.allow_web)

        # tools could be dict (fallback) or list (langchain)
        if isinstance(tools, list):
            tool_names = ", ".join(t.name for t in tools)
            tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        else:
            tool_names = ", ".join(tools.keys())
            tools_block = "\n".join(f"- {k}" for k in tools.keys())

        prompt = REACT_PROMPT.partial(system_prompt=SYSTEM_PROMPT, tools=tools_block, tool_names=tool_names)

        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

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
        if "Final Answer:" in raw:
            return raw.split("Final Answer:", 1)[1].strip()
        # fallback: last non-empty line
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        return lines[-1] if lines else raw.strip()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent. If LangChain executor exists, use it.
        Returns dict with answer, mode, confidence, chunks, trace.
        """
        if self._executor:
            try:
                result = self._executor.invoke({"input": query})
                raw = result.get("output", "") or ""
                final_answer = self._extract_final_answer(raw)
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
                    "answer": f"Agent failed to execute the full plan. Reverting to fallback. Error: {str(e)}",
                    "mode": "agent_error",
                    "confidence": 0.0,
                    "chunks": [],
                    "trace": [{"tool": "agent_failure", "result": str(e)}],
                    "steps": 0
                }

        # Fallback (AgentExecutor not built)
        try:
            out = WrappedLLM().invoke(query)
        except Exception:
            out = "I don’t know based on the available documentation."

        return {
            "answer": out,
            "mode": "agent",
            "confidence": 1.0,
            "chunks": [],
            "trace": [{"tool": "fallback_llm", "result": out}],
            "steps": 1
        }


def build_agent(user_id=None, agent_enabled=True, allow_web: bool = True, **kwargs):
    user_obj = None
    if User:
        try:
            user_obj = User.objects.get(id=user_id)
        except Exception:
            user_obj = None

    return SimpleAgent(user_id=user_id, agent_enabled=agent_enabled, user_obj=user_obj, allow_web=allow_web)
