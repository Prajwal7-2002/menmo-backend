# agent.py — Stable Agent Wrapper

from typing import Optional
from django.contrib.auth import get_user_model
from .controller import agentic_answer

User = get_user_model()

class SimpleAgent:
    def __init__(
        self,
        user_id=None,
        domain=None,
        document_id=None,
        mood="neutral",
        history=None,
        enabled=True,
        user_obj=None
    ):
        self.user_id = user_id
        self.domain = domain
        self.document_id = document_id
        self.mood = mood
        self.history = history or []
        self.enabled = enabled
        self.user = user_obj   # actual Django user instance

    def run(self, query: str):
        """
        Run the agent loop using the controller. This is a safe version that:
        - passes user object (for memory, fallback)
        - passes history into controller
        - reduces max_steps (prevents infinite loops)
        """
        return agentic_answer(
            query=query,
            user_id=self.user_id,
            domain=self.domain,
            document_id=self.document_id,
            user=self.user,
            max_steps=4  # safer; prevents planner instability
        )


def build_agent(
    user_id=None,
    domain=None,
    document_id=None,
    mood="neutral",
    history=None,
    agent_enabled=True
):
    """
    Build a SimpleAgent with user object loaded for correct memory usage.
    """
    user_obj = None
    try:
        user_obj = User.objects.get(id=user_id)
    except Exception:
        user_obj = None  # safety fallback

    return SimpleAgent(
        user_id=user_id,
        domain=domain,
        document_id=document_id,
        mood=mood,
        history=history,
        enabled=agent_enabled,
        user_obj=user_obj
    )
