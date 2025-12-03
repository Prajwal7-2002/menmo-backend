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

RULES:
1. Use tools when needed.
2. Use rag_search for document questions.
3. Use web_search for world knowledge.
4. Use memory_search for user facts.
5. Use rewrite_query if question unclear.
6. Use fallback_llm only when no other tools apply.
7. Your final answer MUST appear only in 'Final Answer:'.
"""


# ---------------------- FULLY VALID LC v0.2+ PROMPT ----------------------

REACT_PROMPT = PromptTemplate.from_template("""
{system_prompt}

You have access to the following tools:
{tools}

The tool names available are:
{tool_names}

Use this format:

Thought: Do I need to use a tool?
Action: <tool_name>
Action Input: <input>

If no tool is needed, respond with:

Final Answer: <your answer>

---

Question: {input}

Thought: Let's think step-by-step.
{agent_scratchpad}
""")



# ------------------------------ AGENT CLASS ------------------------------

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

        tool_names = ", ".join(t.name for t in tools)
        tools_block = "\n".join(f"- {t.name}: {t.description}" for t in tools)

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
            # Increased iterations for stability
            max_iterations=15, 
            max_execution_time=60.0
        )


    def run(self, query: str) -> Dict[str, Any]:
        if self._executor:
            try:
                result = self._executor.invoke({"input": query})
                final_answer = result.get("output", "")

                return {
                    "answer": final_answer,
                    "mode": "agent",
                    "confidence": 1.0,
                    "chunks": [],
                    "trace": [{"tool": "agent", "result": final_answer}],
                    "steps": 1
                }
            except Exception as e:
                logger.exception("Agent error: %s", e)
                # Fallback gracefully instead of crashing
                return {
                    "answer": f"Agent failed to execute the full plan. Reverting to basic LLM. Error: {str(e)}",
                    "mode": "agent_error",
                    "confidence": 0.0,
                    "chunks": [],
                    "trace": [{"tool": "agent_failure", "result": str(e)}],
                    "steps": 0
                }

        # Fallback (If AgentExecutor was not built)
        out = WrappedLLM().invoke(query)
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


import langchain, langchain_core
print("[DEBUG] LangChain Runtime Version:", langchain.__version__)
print("[DEBUG] LangChain-Core Runtime Version:", langchain_core.__version__)
print("[DEBUG] LangChain Path:", langchain.__file__)
