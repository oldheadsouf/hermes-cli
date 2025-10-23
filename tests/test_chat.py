"""Tests for conversational chat session management."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
from hermes_cli.chat import ConversationManager


class TestConversationManagerInitialization:
    """Tests for ConversationManager initialization."""

    def test_init_with_default_directory(self, tmp_path):
        """Test initialization with default conversations directory."""
        with patch('pathlib.Path.home', return_value=tmp_path):
            manager = ConversationManager()
            expected_dir = tmp_path / ".hermes" / "conversations"
            assert manager.conversations_dir == expected_dir
            assert expected_dir.exists()

    def test_init_with_custom_directory(self, tmp_path):
        """Test initialization with custom conversations directory."""
        custom_dir = tmp_path / "custom" / "convos"
        manager = ConversationManager(conversations_dir=custom_dir)
        assert manager.conversations_dir == custom_dir
        assert custom_dir.exists()

    def test_init_creates_directory_if_not_exists(self, tmp_path):
        """Test that initialization creates directory structure."""
        conv_dir = tmp_path / "new" / "nested" / "dir"
        assert not conv_dir.exists()
        manager = ConversationManager(conversations_dir=conv_dir)
        assert conv_dir.exists()

    def test_active_session_file_location(self, tmp_path):
        """Test that active session file is in parent directory."""
        conv_dir = tmp_path / "conversations"
        manager = ConversationManager(conversations_dir=conv_dir)
        expected_active_file = conv_dir.parent / ".active_session"
        assert manager.active_session_file == expected_active_file


class TestGetConversationPath:
    """Tests for _get_conversation_path method."""

    def test_get_conversation_path_simple_name(self, tmp_path):
        """Test getting path for simple conversation name."""
        manager = ConversationManager(conversations_dir=tmp_path)
        path = manager._get_conversation_path("myconvo")
        assert path == tmp_path / "myconvo.json"

    def test_get_conversation_path_sanitizes_special_chars(self, tmp_path):
        """Test that special characters are sanitized."""
        manager = ConversationManager(conversations_dir=tmp_path)
        path = manager._get_conversation_path("my/conversation?name!")
        assert path == tmp_path / "my_conversation_name_.json"

    def test_get_conversation_path_allows_hyphens_underscores(self, tmp_path):
        """Test that hyphens and underscores are preserved."""
        manager = ConversationManager(conversations_dir=tmp_path)
        path = manager._get_conversation_path("my-convo_name")
        assert path == tmp_path / "my-convo_name.json"

    def test_get_conversation_path_sanitizes_spaces(self, tmp_path):
        """Test that spaces are converted to underscores."""
        manager = ConversationManager(conversations_dir=tmp_path)
        path = manager._get_conversation_path("my conversation")
        assert path == tmp_path / "my_conversation.json"


class TestEnsureUniqueName:
    """Tests for _ensure_unique_name method."""

    def test_ensure_unique_name_when_no_conflict(self, tmp_path):
        """Test unique name when no existing conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        unique_name = manager._ensure_unique_name("newconvo")
        assert unique_name == "newconvo"

    def test_ensure_unique_name_increments_on_conflict(self, tmp_path):
        """Test name increments when conversation exists."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create existing conversation file
        (tmp_path / "test.json").touch()

        unique_name = manager._ensure_unique_name("test")
        assert unique_name == "test-2"

    def test_ensure_unique_name_multiple_conflicts(self, tmp_path):
        """Test name increments correctly with multiple conflicts."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create multiple existing files
        (tmp_path / "chat.json").touch()
        (tmp_path / "chat-2.json").touch()
        (tmp_path / "chat-3.json").touch()

        unique_name = manager._ensure_unique_name("chat")
        assert unique_name == "chat-4"


class TestCreateConversation:
    """Tests for create_conversation method."""

    def test_create_conversation_basic(self, tmp_path):
        """Test creating a basic conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)
        initial_msg = {"role": "user", "content": "Hello"}

        name, path = manager.create_conversation(
            name="test-chat",
            initial_message=initial_msg
        )

        assert name == "test-chat"
        assert path.exists()
        assert path == tmp_path / "test-chat.json"

        # Verify file contents
        with open(path, 'r') as f:
            data = json.load(f)

        assert data["name"] == "test-chat"
        assert data["model"] == "Hermes-4-405B"
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 2048
        assert data["schema"] is None
        assert len(data["messages"]) == 1
        assert data["messages"][0] == initial_msg

    def test_create_conversation_with_system_prompt(self, tmp_path):
        """Test creating conversation with system prompt."""
        manager = ConversationManager(conversations_dir=tmp_path)
        initial_msg = {"role": "user", "content": "Hello"}

        name, path = manager.create_conversation(
            name="chat",
            initial_message=initial_msg,
            system_prompt="You are helpful"
        )

        with open(path, 'r') as f:
            data = json.load(f)

        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "system"
        assert data["messages"][0]["content"] == "You are helpful"
        assert data["messages"][1] == initial_msg

    def test_create_conversation_custom_parameters(self, tmp_path):
        """Test creating conversation with custom parameters."""
        manager = ConversationManager(conversations_dir=tmp_path)
        initial_msg = {"role": "user", "content": "Test"}
        schema = {"type": "object"}

        name, path = manager.create_conversation(
            name="custom",
            initial_message=initial_msg,
            model="Hermes-4-70B",
            temperature=0.5,
            max_tokens=1024,
            schema=schema
        )

        with open(path, 'r') as f:
            data = json.load(f)

        assert data["model"] == "Hermes-4-70B"
        assert data["temperature"] == 0.5
        assert data["max_tokens"] == 1024
        assert data["schema"] == schema

    def test_create_conversation_has_timestamps(self, tmp_path):
        """Test that created conversation has timestamp fields."""
        manager = ConversationManager(conversations_dir=tmp_path)
        initial_msg = {"role": "user", "content": "Hi"}

        before = datetime.now()
        name, path = manager.create_conversation(
            name="timestamped",
            initial_message=initial_msg
        )
        after = datetime.now()

        with open(path, 'r') as f:
            data = json.load(f)

        assert "created_at" in data
        assert "updated_at" in data

        # Verify timestamps are valid ISO format
        created = datetime.fromisoformat(data["created_at"])
        updated = datetime.fromisoformat(data["updated_at"])

        assert before <= created <= after
        assert before <= updated <= after

    def test_create_conversation_auto_increments_name(self, tmp_path):
        """Test that duplicate names are auto-incremented."""
        manager = ConversationManager(conversations_dir=tmp_path)
        msg1 = {"role": "user", "content": "First"}
        msg2 = {"role": "user", "content": "Second"}

        name1, path1 = manager.create_conversation("duplicate", msg1)
        name2, path2 = manager.create_conversation("duplicate", msg2)

        assert name1 == "duplicate"
        assert name2 == "duplicate-2"
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()


class TestLoadConversation:
    """Tests for load_conversation method."""

    def test_load_conversation_success(self, tmp_path):
        """Test loading an existing conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create a conversation first
        initial_msg = {"role": "user", "content": "Hello"}
        name, _ = manager.create_conversation("loadtest", initial_msg)

        # Load it back
        data = manager.load_conversation("loadtest")

        assert data["name"] == "loadtest"
        assert data["messages"][0] == initial_msg

    def test_load_conversation_not_found(self, tmp_path):
        """Test loading non-existent conversation raises error."""
        manager = ConversationManager(conversations_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            manager.load_conversation("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_load_conversation_with_sanitized_name(self, tmp_path):
        """Test loading conversation with sanitized name."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create conversation with special chars
        initial_msg = {"role": "user", "content": "Test"}
        # The create_conversation stores the original name, not sanitized version
        name, _ = manager.create_conversation("my/chat!", initial_msg)

        # Load using same name (sanitization happens internally)
        data = manager.load_conversation("my/chat!")
        # The stored name should be the original, not sanitized
        assert data["name"] == "my/chat!"


class TestSaveConversation:
    """Tests for save_conversation method."""

    def test_save_conversation_updates_file(self, tmp_path):
        """Test that save_conversation writes to file."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create initial conversation
        initial_msg = {"role": "user", "content": "Hi"}
        name, path = manager.create_conversation("savetest", initial_msg)

        # Load and modify
        data = manager.load_conversation(name)
        data["messages"].append({"role": "assistant", "content": "Hello!"})

        # Save modified data
        manager.save_conversation(name, data)

        # Reload and verify
        reloaded = manager.load_conversation(name)
        assert len(reloaded["messages"]) == 2
        assert reloaded["messages"][1]["content"] == "Hello!"

    def test_save_conversation_updates_timestamp(self, tmp_path):
        """Test that save_conversation updates the updated_at timestamp."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create conversation
        initial_msg = {"role": "user", "content": "Hi"}
        name, _ = manager.create_conversation("timestamp-test", initial_msg)

        # Get original timestamp
        original_data = manager.load_conversation(name)
        original_updated = original_data["updated_at"]

        # Wait a tiny bit and save
        import time
        time.sleep(0.01)

        # Modify and save
        original_data["messages"].append({"role": "assistant", "content": "Hi!"})
        manager.save_conversation(name, original_data)

        # Verify timestamp was updated
        updated_data = manager.load_conversation(name)
        assert updated_data["updated_at"] > original_updated


class TestAddMessage:
    """Tests for add_message method."""

    def test_add_message_user(self, tmp_path):
        """Test adding a user message."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create conversation
        initial_msg = {"role": "user", "content": "First"}
        name, _ = manager.create_conversation("msgtest", initial_msg)

        # Add message
        manager.add_message(name, "user", "Second message")

        # Verify
        data = manager.load_conversation(name)
        assert len(data["messages"]) == 2
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"] == "Second message"

    def test_add_message_assistant(self, tmp_path):
        """Test adding an assistant message."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create conversation
        initial_msg = {"role": "user", "content": "Hello"}
        name, _ = manager.create_conversation("assistant-test", initial_msg)

        # Add assistant response
        manager.add_message(name, "assistant", "Hi there!")

        # Verify
        data = manager.load_conversation(name)
        assert len(data["messages"]) == 2
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"] == "Hi there!"

    def test_add_message_multiple(self, tmp_path):
        """Test adding multiple messages in sequence."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create conversation
        initial_msg = {"role": "user", "content": "Start"}
        name, _ = manager.create_conversation("multi", initial_msg)

        # Add multiple messages
        manager.add_message(name, "assistant", "Response 1")
        manager.add_message(name, "user", "Follow up")
        manager.add_message(name, "assistant", "Response 2")

        # Verify
        data = manager.load_conversation(name)
        assert len(data["messages"]) == 4
        assert data["messages"][1]["content"] == "Response 1"
        assert data["messages"][2]["content"] == "Follow up"
        assert data["messages"][3]["content"] == "Response 2"


class TestGetMessages:
    """Tests for get_messages method."""

    def test_get_messages_returns_all_messages(self, tmp_path):
        """Test that get_messages returns all conversation messages."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create conversation with system prompt
        initial_msg = {"role": "user", "content": "Hello"}
        name, _ = manager.create_conversation(
            "getmsg",
            initial_msg,
            system_prompt="Be helpful"
        )

        # Add more messages
        manager.add_message(name, "assistant", "Hi!")
        manager.add_message(name, "user", "How are you?")

        # Get messages
        messages = manager.get_messages(name)

        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"

    def test_get_messages_returns_list(self, tmp_path):
        """Test that get_messages returns a list."""
        manager = ConversationManager(conversations_dir=tmp_path)

        initial_msg = {"role": "user", "content": "Test"}
        name, _ = manager.create_conversation("listtest", initial_msg)

        messages = manager.get_messages(name)
        assert isinstance(messages, list)


class TestListConversations:
    """Tests for list_conversations method."""

    def test_list_conversations_empty(self, tmp_path):
        """Test listing conversations when none exist."""
        manager = ConversationManager(conversations_dir=tmp_path)
        conversations = manager.list_conversations()
        assert conversations == []

    def test_list_conversations_single(self, tmp_path):
        """Test listing a single conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)

        initial_msg = {"role": "user", "content": "Hello"}
        manager.create_conversation("single", initial_msg)

        conversations = manager.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["name"] == "single"
        assert conversations[0]["model"] == "Hermes-4-405B"
        assert conversations[0]["message_count"] == 1

    def test_list_conversations_multiple(self, tmp_path):
        """Test listing multiple conversations."""
        manager = ConversationManager(conversations_dir=tmp_path)

        msg = {"role": "user", "content": "Test"}
        manager.create_conversation("first", msg)
        manager.create_conversation("second", msg)
        manager.create_conversation("third", msg)

        conversations = manager.list_conversations()
        assert len(conversations) == 3

        names = [c["name"] for c in conversations]
        assert "first" in names
        assert "second" in names
        assert "third" in names

    def test_list_conversations_sorted_by_updated_at(self, tmp_path):
        """Test that conversations are sorted by most recently updated."""
        manager = ConversationManager(conversations_dir=tmp_path)

        import time

        msg = {"role": "user", "content": "Test"}
        manager.create_conversation("oldest", msg)
        time.sleep(0.01)
        manager.create_conversation("middle", msg)
        time.sleep(0.01)
        name3, _ = manager.create_conversation("newest", msg)

        conversations = manager.list_conversations()

        # Most recent should be first
        assert conversations[0]["name"] == "newest"
        assert conversations[2]["name"] == "oldest"

    def test_list_conversations_includes_metadata(self, tmp_path):
        """Test that listed conversations include all metadata."""
        manager = ConversationManager(conversations_dir=tmp_path)

        msg = {"role": "user", "content": "Hello"}
        manager.create_conversation(
            "meta",
            msg,
            model="Hermes-4-70B",
            system_prompt="System"
        )

        conversations = manager.list_conversations()
        conv = conversations[0]

        assert "name" in conv
        assert "created_at" in conv
        assert "updated_at" in conv
        assert "model" in conv
        assert "message_count" in conv
        assert conv["model"] == "Hermes-4-70B"
        assert conv["message_count"] == 2  # system + user

    def test_list_conversations_skips_malformed_files(self, tmp_path):
        """Test that malformed JSON files are skipped."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create valid conversation
        msg = {"role": "user", "content": "Valid"}
        manager.create_conversation("valid", msg)

        # Create malformed JSON file
        malformed_path = tmp_path / "malformed.json"
        with open(malformed_path, 'w') as f:
            f.write("{invalid json content")

        # Should only return valid conversation
        conversations = manager.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["name"] == "valid"


class TestActiveSession:
    """Tests for active session management."""

    def test_set_and_get_active_session(self, tmp_path):
        """Test setting and getting active session."""
        manager = ConversationManager(conversations_dir=tmp_path)

        manager.set_active_session("my-chat")
        active = manager.get_active_session()

        assert active == "my-chat"

    def test_get_active_session_when_none(self, tmp_path):
        """Test getting active session when none is set."""
        # Use a completely fresh tmp_path to avoid test pollution
        fresh_dir = tmp_path / "fresh_test"
        fresh_dir.mkdir()
        manager = ConversationManager(conversations_dir=fresh_dir)
        active = manager.get_active_session()
        assert active is None

    def test_clear_active_session(self, tmp_path):
        """Test clearing the active session."""
        manager = ConversationManager(conversations_dir=tmp_path)

        manager.set_active_session("test-chat")
        assert manager.get_active_session() == "test-chat"

        manager.clear_active_session()
        assert manager.get_active_session() is None

    def test_clear_active_session_when_none(self, tmp_path):
        """Test that clearing non-existent active session doesn't error."""
        manager = ConversationManager(conversations_dir=tmp_path)
        manager.clear_active_session()  # Should not raise
        assert manager.get_active_session() is None

    def test_set_active_session_overwrites(self, tmp_path):
        """Test that setting active session overwrites previous."""
        manager = ConversationManager(conversations_dir=tmp_path)

        manager.set_active_session("first")
        manager.set_active_session("second")

        assert manager.get_active_session() == "second"

    def test_get_active_session_handles_corrupted_file(self, tmp_path):
        """Test that corrupted active session file returns None."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create active session file
        manager.set_active_session("test")

        # Corrupt it by making it unreadable (simulate permission error)
        # Instead, we'll just test the exception handling path
        with patch('builtins.open', side_effect=Exception("Read error")):
            active = manager.get_active_session()
            assert active is None


class TestDeleteConversation:
    """Tests for delete_conversation method."""

    def test_delete_conversation_success(self, tmp_path):
        """Test deleting an existing conversation."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create conversation
        msg = {"role": "user", "content": "Test"}
        name, path = manager.create_conversation("todelete", msg)

        assert path.exists()

        # Delete it
        manager.delete_conversation(name)

        assert not path.exists()

    def test_delete_conversation_not_found(self, tmp_path):
        """Test deleting non-existent conversation raises error."""
        manager = ConversationManager(conversations_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            manager.delete_conversation("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_delete_conversation_clears_active_session(self, tmp_path):
        """Test that deleting active conversation clears active session."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create and set as active
        msg = {"role": "user", "content": "Test"}
        name, _ = manager.create_conversation("active-delete", msg)
        manager.set_active_session(name)

        assert manager.get_active_session() == name

        # Delete conversation
        manager.delete_conversation(name)

        # Active session should be cleared
        assert manager.get_active_session() is None

    def test_delete_conversation_preserves_other_active_session(self, tmp_path):
        """Test that deleting non-active conversation preserves active session."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create two conversations
        msg = {"role": "user", "content": "Test"}
        name1, _ = manager.create_conversation("keep", msg)
        name2, _ = manager.create_conversation("delete", msg)

        # Set first as active
        manager.set_active_session(name1)

        # Delete second
        manager.delete_conversation(name2)

        # Active session should still be first
        assert manager.get_active_session() == name1


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_conversation_name(self, tmp_path):
        """Test handling of empty conversation name."""
        manager = ConversationManager(conversations_dir=tmp_path)
        msg = {"role": "user", "content": "Test"}

        # Empty string should create a file
        name, path = manager.create_conversation("", msg)
        assert path.exists()

    def test_very_long_conversation_name(self, tmp_path):
        """Test handling of very long conversation name."""
        manager = ConversationManager(conversations_dir=tmp_path)
        msg = {"role": "user", "content": "Test"}

        # Use a long but reasonable name (filesystem limit is typically 255 chars)
        # Account for the ".json" extension (4 chars)
        long_name = "a" * 200
        name, path = manager.create_conversation(long_name, msg)

        # Should create successfully
        assert path.exists()
        data = manager.load_conversation(long_name)
        assert data["messages"][0] == msg

    def test_unicode_conversation_name(self, tmp_path):
        """Test handling of unicode characters in name."""
        manager = ConversationManager(conversations_dir=tmp_path)
        msg = {"role": "user", "content": "Test"}

        unicode_name = "chat-日本語-emoji-🚀"
        name, path = manager.create_conversation(unicode_name, msg)

        assert path.exists()
        data = manager.load_conversation(unicode_name)
        assert len(data["messages"]) == 1

    def test_conversation_with_empty_messages_list(self, tmp_path):
        """Test handling conversation data with empty messages."""
        manager = ConversationManager(conversations_dir=tmp_path)

        # Create valid conversation
        msg = {"role": "user", "content": "Test"}
        name, path = manager.create_conversation("empty-test", msg)

        # Manually modify to have empty messages
        data = manager.load_conversation(name)
        data["messages"] = []
        manager.save_conversation(name, data)

        # Should still work
        messages = manager.get_messages(name)
        assert messages == []

    def test_add_message_to_nonexistent_conversation(self, tmp_path):
        """Test adding message to non-existent conversation raises error."""
        manager = ConversationManager(conversations_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            manager.add_message("nonexistent", "user", "Hello")
