from typing import Optional
from .controller import agentic_answer

class SimpleAgent:
    def __init__(self, user_id=None, domain=None, document_id=None, mood="neutral", history=None, enabled=False):
        self.user_id = user_id
        self.domain = domain
        self.document_id = document_id
        self.mood = mood
        self.history = history or []
        self.enabled = enabled

    def run(self, query: str):
        return agentic_answer(
            query=query,
            user_id=self.user_id,
            domain=self.domain,
            document_id=self.document_id,
            user=None,
            max_steps=4
        )

def build_agent(user_id=None, domain=None, document_id=None, mood="neutral", history=None, agent_enabled=False):
    return SimpleAgent(user_id=user_id, domain=domain, document_id=document_id, mood=mood, history=history, enabled=agent_enabled)
