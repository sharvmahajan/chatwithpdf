from typing import List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Session-based chat memory.
    Sliding window: keep last N turns.
    Persistent in-memory (extensible to Redis).
    """

    def __init__(self, max_turns: int = 6):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_turns = max_turns

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a turn to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Enforce sliding window (N turns * 2 roles)
        if len(self.sessions[session_id]) > self.max_turns * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_turns * 2:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve core history for LLM context."""
        return self.sessions.get(session_id, [])

    def clear(self, session_id: str):
        """Reset session memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]

memory_service = MemoryService()
