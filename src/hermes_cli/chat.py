"""Conversational chat session management."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class ConversationManager:
    """Manages conversational chat sessions with local JSON persistence."""

    def __init__(self, conversations_dir: Optional[Path] = None):
        """Initialize the conversation manager.

        Args:
            conversations_dir: Directory to store conversation files.
                             Defaults to ~/.hermes/conversations
        """
        if conversations_dir is None:
            conversations_dir = Path.home() / ".hermes" / "conversations"

        self.conversations_dir = conversations_dir
        self.active_session_file = self.conversations_dir.parent / ".active_session"

        # Create conversations directory if it doesn't exist
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def _get_conversation_path(self, name: str) -> Path:
        """Get the file path for a conversation.

        Args:
            name: Name of the conversation

        Returns:
            Path to the conversation JSON file
        """
        # Sanitize the name to be filesystem-safe
        safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
        return self.conversations_dir / f"{safe_name}.json"

    def _ensure_unique_name(self, name: str) -> str:
        """Ensure the conversation name is unique, auto-incrementing if needed.

        Args:
            name: Desired conversation name

        Returns:
            Unique conversation name (may have -2, -3, etc. appended)
        """
        original_name = name
        counter = 2

        while self._get_conversation_path(name).exists():
            name = f"{original_name}-{counter}"
            counter += 1

        return name

    def create_conversation(
        self,
        name: str,
        initial_message: Dict[str, str],
        system_prompt: Optional[str] = None,
        model: str = "Hermes-4-405B",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        schema: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Path]:
        """Create a new conversation session.

        Args:
            name: Name for the conversation
            initial_message: The first user message dict
            system_prompt: Optional system prompt
            model: Model to use for this conversation
            temperature: Temperature setting
            max_tokens: Max tokens setting
            schema: Optional JSON schema

        Returns:
            Tuple of (actual_name_used, conversation_file_path)
        """
        # Ensure unique name
        unique_name = self._ensure_unique_name(name)

        # Build initial messages array
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(initial_message)

        # Create conversation metadata
        conversation_data = {
            "name": unique_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "schema": schema,
            "messages": messages
        }

        # Save to file
        conv_path = self._get_conversation_path(unique_name)
        with open(conv_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2)

        return unique_name, conv_path

    def load_conversation(self, name: str) -> Dict[str, Any]:
        """Load an existing conversation.

        Args:
            name: Name of the conversation to load

        Returns:
            Conversation data dictionary

        Raises:
            FileNotFoundError: If conversation doesn't exist
        """
        conv_path = self._get_conversation_path(name)

        if not conv_path.exists():
            raise FileNotFoundError(
                f"Conversation '{name}' not found.\n"
                f"Use 'hermes chat --name <name>' to create a new conversation."
            )

        with open(conv_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_conversation(self, name: str, conversation_data: Dict[str, Any]) -> None:
        """Save conversation data to file.

        Args:
            name: Name of the conversation
            conversation_data: Full conversation data to save
        """
        conv_path = self._get_conversation_path(name)

        # Update timestamp
        conversation_data["updated_at"] = datetime.now().isoformat()

        with open(conv_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2)

    def add_message(
        self,
        name: str,
        role: str,
        content: str
    ) -> None:
        """Add a message to an existing conversation.

        Args:
            name: Name of the conversation
            role: Message role (user or assistant)
            content: Message content
        """
        conversation_data = self.load_conversation(name)
        conversation_data["messages"].append({
            "role": role,
            "content": content
        })
        self.save_conversation(name, conversation_data)

    def get_messages(self, name: str) -> List[Dict[str, str]]:
        """Get all messages from a conversation.

        Args:
            name: Name of the conversation

        Returns:
            List of message dictionaries
        """
        conversation_data = self.load_conversation(name)
        return conversation_data["messages"]

    def list_conversations(self) -> List[Dict[str, str]]:
        """List all available conversations.

        Returns:
            List of dicts with conversation metadata
        """
        conversations = []

        for conv_file in self.conversations_dir.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    conversations.append({
                        "name": data.get("name", conv_file.stem),
                        "created_at": data.get("created_at", "Unknown"),
                        "updated_at": data.get("updated_at", "Unknown"),
                        "model": data.get("model", "Unknown"),
                        "message_count": len(data.get("messages", []))
                    })
            except (json.JSONDecodeError, KeyError):
                # Skip malformed conversation files
                continue

        # Sort by most recently updated
        conversations.sort(key=lambda x: x["updated_at"], reverse=True)
        return conversations

    def set_active_session(self, name: str) -> None:
        """Set the active conversation session.

        Args:
            name: Name of the conversation to set as active
        """
        with open(self.active_session_file, 'w', encoding='utf-8') as f:
            f.write(name)

    def get_active_session(self) -> Optional[str]:
        """Get the currently active conversation session.

        Returns:
            Name of active conversation, or None if no active session
        """
        if not self.active_session_file.exists():
            return None

        try:
            with open(self.active_session_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return None

    def clear_active_session(self) -> None:
        """Clear the active conversation session."""
        if self.active_session_file.exists():
            self.active_session_file.unlink()

    def delete_conversation(self, name: str) -> None:
        """Delete a conversation.

        Args:
            name: Name of the conversation to delete

        Raises:
            FileNotFoundError: If conversation doesn't exist
        """
        conv_path = self._get_conversation_path(name)

        if not conv_path.exists():
            raise FileNotFoundError(f"Conversation '{name}' not found.")

        conv_path.unlink()

        # If this was the active session, clear it
        if self.get_active_session() == name:
            self.clear_active_session()
