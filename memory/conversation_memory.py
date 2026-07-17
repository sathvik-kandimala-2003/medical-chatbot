"""In-memory, per-session conversation history.

Built on langchain_core's InMemoryChatMessageHistory rather than
langchain's ConversationBufferMemory: LangChain deprecated
ConversationBufferMemory in 0.3.1 (removal planned for 2.0) in favor of
managing chat messages directly, so building new code on it would mean
starting production-grade work on an API that's already being phased out.
InMemoryChatMessageHistory gives the same "list of turns" model on the
currently maintained primitive.

One ConversationMemory instance belongs to a single Streamlit session
(st.session_state) - never to st.cache_resource, which is shared across
every user of the deployed app and would leak one user's chat history into
another's.
"""

import logging
from typing import List

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# Caps how many past turns feed into query condensation and the answer
# prompt, so a long session's context doesn't grow unbounded.
DEFAULT_MAX_TURNS = 6


class ConversationMemory:
    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        self._history = InMemoryChatMessageHistory()
        self._max_turns = max_turns

    def add_user_message(self, text: str) -> None:
        self._history.add_user_message(text)

    def add_ai_message(self, text: str) -> None:
        self._history.add_ai_message(text)

    def is_empty(self) -> bool:
        return len(self._history.messages) == 0

    def recent_messages(self) -> List[BaseMessage]:
        return self._history.messages[-(self._max_turns * 2):]

    def as_text(self) -> str:
        """Format recent turns as "User: ...\\nAssistant: ..." for prompts."""
        lines = []
        for message in self.recent_messages():
            role = "User" if isinstance(message, HumanMessage) else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()
