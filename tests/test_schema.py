"""Tests for schema handling."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from hermes_cli.schema import (
    is_json_string,
    validate_schema_structure,
    load_schema,
    build_system_prompt_with_schema,
    should_disable_streaming
)


class TestIsJsonString:
    """Tests for is_json_string function."""

    def test_identifies_json_object(self):
        """Test that JSON objects starting with { are identified."""
        assert is_json_string('{"type": "object"}') is True
        assert is_json_string(' {"type": "object"}') is True
        assert is_json_string('\n{"type": "object"}') is True

    def test_identifies_json_array(self):
        """Test that JSON arrays starting with [ are identified."""
        assert is_json_string('[1, 2, 3]') is True
        assert is_json_string(' [1, 2, 3]') is True
        assert is_json_string('\n[{"type": "string"}]') is True

    def test_identifies_file_paths(self):
        """Test that file paths are correctly identified as not JSON."""
        assert is_json_string('schema.json') is False
        assert is_json_string('./schemas/response.json') is False
        assert is_json_string('/path/to/schema.json') is False
        assert is_json_string('~/config/schema.json') is False

    def test_handles_whitespace(self):
        """Test proper handling of whitespace."""
        assert is_json_string('   {"key": "value"}') is True
        assert is_json_string('\t\n{"key": "value"}') is True
        assert is_json_string('   file.json') is False

    def test_edge_cases(self):
        """Test edge cases."""
        assert is_json_string('') is False
        assert is_json_string('   ') is False
        assert is_json_string('not-json-or-path') is False


class TestValidateSchemaStructure:
    """Tests for validate_schema_structure function."""

    def test_valid_schema_with_type(self):
        """Test validation of valid schema with type field."""
        schema = {"type": "object", "properties": {}}
        validate_schema_structure(schema)  # Should not raise

    def test_valid_schema_without_type(self):
        """Test validation of schema without type field (allowed)."""
        schema = {"properties": {"name": {"type": "string"}}}
        validate_schema_structure(schema)  # Should not raise

    def test_empty_schema_raises_error(self):
        """Test that empty schema raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_schema_structure({})
        assert "cannot be empty" in str(exc_info.value)

    def test_non_dict_schema_raises_error(self):
        """Test that non-dictionary schemas raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_schema_structure([])
        assert "must be a JSON object" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            validate_schema_structure("string")
        assert "must be a JSON object" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            validate_schema_structure(None)
        assert "must be a JSON object" in str(exc_info.value)

    def test_complex_valid_schema(self):
        """Test validation of complex valid schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "email"]
        }
        validate_schema_structure(schema)  # Should not raise


class TestLoadSchema:
    """Tests for load_schema function."""

    def test_load_schema_from_json_string(self):
        """Test loading schema from JSON string."""
        json_string = '{"type": "object", "properties": {"key": {"type": "string"}}}'
        schema = load_schema(json_string)

        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_load_schema_from_json_array(self):
        """Test loading schema that is a JSON array (should fail validation)."""
        json_string = '[{"type": "string"}]'

        # Arrays are not valid schemas, should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            load_schema(json_string)

        assert "must be a JSON object" in str(exc_info.value)

    def test_load_schema_with_whitespace_in_string(self):
        """Test loading schema with whitespace padding."""
        json_string = '  \n  {"type": "object"}  \n  '
        schema = load_schema(json_string)

        assert isinstance(schema, dict)
        assert schema["type"] == "object"

    def test_load_schema_from_file(self, tmp_path):
        """Test loading schema from file."""
        schema_content = {"type": "object", "properties": {"answer": {"type": "string"}}}
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema_content))

        schema = load_schema(str(schema_file))

        assert schema == schema_content

    def test_load_schema_from_file_with_expanduser(self, tmp_path, monkeypatch):
        """Test loading schema from file with ~ expansion."""
        schema_content = {"type": "object"}
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema_content))

        # Mock expanduser to return our tmp_path
        def mock_expanduser(self):
            if str(self).startswith("~"):
                return tmp_path / str(self)[2:]  # Remove ~/ prefix
            return self

        with patch.object(Path, 'expanduser', mock_expanduser):
            schema = load_schema("~/schema.json")
            assert schema == schema_content

    def test_load_schema_file_not_found(self):
        """Test error handling when schema file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_schema("nonexistent_schema.json")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "nonexistent_schema.json" in error_msg

    def test_load_schema_invalid_json_string(self):
        """Test error handling for invalid JSON string."""
        invalid_json = '{"type": "object", invalid}'

        with pytest.raises(ValueError) as exc_info:
            load_schema(invalid_json)

        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg
        assert "Example:" in error_msg

    def test_load_schema_invalid_json_in_file(self, tmp_path):
        """Test error handling for invalid JSON in file."""
        schema_file = tmp_path / "bad_schema.json"
        schema_file.write_text('{"type": "object", invalid}')

        with pytest.raises(ValueError) as exc_info:
            load_schema(str(schema_file))

        error_msg = str(exc_info.value)
        assert "Invalid JSON in schema file" in error_msg
        assert "bad_schema.json" in error_msg

    def test_load_schema_empty_input(self):
        """Test error handling for empty input."""
        with pytest.raises(ValueError) as exc_info:
            load_schema("")
        assert "cannot be empty" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            load_schema("   ")
        assert "cannot be empty" in str(exc_info.value)

    def test_load_schema_empty_schema_raises_error(self):
        """Test that loading empty schema dict raises error."""
        with pytest.raises(ValueError) as exc_info:
            load_schema('{}')
        assert "cannot be empty" in str(exc_info.value)

    def test_load_schema_path_is_directory(self, tmp_path):
        """Test error handling when path is a directory."""
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()

        with pytest.raises(ValueError) as exc_info:
            load_schema(str(schema_dir))

        error_msg = str(exc_info.value)
        assert "not a file" in error_msg

    def test_load_schema_permission_denied(self, tmp_path):
        """Test error handling for permission denied."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "object", "properties": {}}')
        schema_file.chmod(0o000)  # Remove all permissions

        try:
            with pytest.raises(ValueError) as exc_info:
                load_schema(str(schema_file))

            error_msg = str(exc_info.value)
            assert "Permission denied" in error_msg or "Error reading" in error_msg
        finally:
            # Restore permissions for cleanup
            schema_file.chmod(0o644)

    def test_load_complex_schema_from_string(self):
        """Test loading complex schema from string."""
        complex_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string", "format": "date-time"}
                    }
                }
            },
            "required": ["name"]
        }

        json_string = json.dumps(complex_schema)
        schema = load_schema(json_string)

        assert schema == complex_schema

    def test_load_schema_with_unicode(self, tmp_path):
        """Test loading schema with unicode characters."""
        schema_content = {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Hello 世界 🌍"}
            }
        }
        schema_file = tmp_path / "unicode_schema.json"
        schema_file.write_text(json.dumps(schema_content, ensure_ascii=False), encoding='utf-8')

        schema = load_schema(str(schema_file))

        assert schema == schema_content
        assert "Hello 世界 🌍" in schema["properties"]["message"]["description"]


class TestBuildSystemPromptWithSchema:
    """Tests for build_system_prompt_with_schema function."""

    def test_build_prompt_with_user_system_prompt(self):
        """Test building system prompt with user-provided system prompt."""
        user_prompt = "You are a helpful assistant"
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        result = build_system_prompt_with_schema(user_prompt, schema)

        assert user_prompt in result
        assert "You must respond with valid JSON matching this schema:" in result
        assert '"type": "object"' in result
        assert '"properties"' in result

    def test_build_prompt_without_user_system_prompt(self):
        """Test building system prompt without user-provided prompt."""
        schema = {"type": "object", "properties": {"result": {"type": "integer"}}}

        result = build_system_prompt_with_schema(None, schema)

        assert result.startswith("You must respond with valid JSON")
        assert not result.startswith("\n")  # Should not have leading newlines
        assert '"type": "object"' in result

    def test_build_prompt_with_empty_user_prompt(self):
        """Test building system prompt with empty string user prompt."""
        schema = {"type": "string"}

        result = build_system_prompt_with_schema("", schema)

        # Empty string is falsy, so should be treated like None
        assert result.startswith("You must respond with valid JSON")

    def test_build_prompt_schema_formatting(self):
        """Test that schema is formatted with indentation for readability."""
        schema = {
            "type": "object",
            "properties": {
                "key1": {"type": "string"},
                "key2": {"type": "integer"}
            }
        }

        result = build_system_prompt_with_schema(None, schema)

        # Check that it's indented (not compact)
        assert '  "type":' in result or '  "properties":' in result
        formatted_schema = json.dumps(schema, indent=2)
        assert formatted_schema in result

    def test_build_prompt_preserves_user_prompt_exactly(self):
        """Test that user prompt is preserved exactly as provided."""
        user_prompt = "Be concise.\nUse formal language.\nNo explanations."
        schema = {"type": "boolean"}

        result = build_system_prompt_with_schema(user_prompt, schema)

        assert result.startswith(user_prompt)
        assert user_prompt in result

    def test_build_prompt_with_complex_schema(self):
        """Test building prompt with complex nested schema."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["users"]
        }

        result = build_system_prompt_with_schema("Return user data", schema)

        assert "Return user data" in result
        assert '"users"' in result
        assert '"items"' in result
        assert '"required"' in result

    def test_build_prompt_with_special_characters_in_user_prompt(self):
        """Test building prompt with special characters in user prompt."""
        user_prompt = "Answer questions about C++ & Python!!! Use @mentions."
        schema = {"type": "string"}

        result = build_system_prompt_with_schema(user_prompt, schema)

        assert user_prompt in result
        assert "C++ & Python!!!" in result
        assert "@mentions" in result

    def test_build_prompt_with_unicode_in_schema(self):
        """Test building prompt with unicode characters in schema."""
        schema = {
            "type": "object",
            "properties": {
                "greeting": {
                    "type": "string",
                    "description": "Say hello in different languages: 你好, مرحبا, שלום"
                }
            }
        }

        result = build_system_prompt_with_schema(None, schema)

        # json.dumps() may encode unicode as escape sequences (\uXXXX)
        # Check that either the unicode characters or their escape sequences are present
        assert ("你好" in result or "\\u4f60\\u597d" in result)
        assert ("مرحبا" in result or "\\u0645\\u0631\\u062d\\u0628\\u0627" in result)
        assert ("שלום" in result or "\\u05e9\\u05dc\\u05d5\\u05dd" in result)

    def test_build_prompt_concatenation_format(self):
        """Test the exact concatenation format matches CLAUDE.md spec."""
        user_prompt = "You are helpful"
        schema = {"type": "object"}

        result = build_system_prompt_with_schema(user_prompt, schema)

        # According to CLAUDE.md: {user_system_prompt}\n\nYou must respond...
        assert "\n\n" in result
        parts = result.split("\n\n", 1)
        assert parts[0] == user_prompt
        assert parts[1].startswith("You must respond with valid JSON")


class TestShouldDisableStreaming:
    """Tests for should_disable_streaming function."""

    def test_disable_streaming_when_schema_provided(self):
        """Test that streaming is disabled when schema is provided."""
        schema = {"type": "object", "properties": {}}
        assert should_disable_streaming(schema) is True

    def test_enable_streaming_when_no_schema(self):
        """Test that streaming is enabled when no schema is provided."""
        assert should_disable_streaming(None) is False

    def test_disable_streaming_with_complex_schema(self):
        """Test streaming disabled with complex schema."""
        schema = {
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "string"}}
            }
        }
        assert should_disable_streaming(schema) is True

    def test_disable_streaming_with_simple_schema(self):
        """Test streaming disabled even with simple schema."""
        schema = {"type": "string"}
        assert should_disable_streaming(schema) is True

    def test_disable_streaming_with_empty_schema(self):
        """Test streaming disabled even with empty dict (though invalid)."""
        # Empty dict is still a schema object, so should disable streaming
        schema = {}
        assert should_disable_streaming(schema) is True


class TestIntegration:
    """Integration tests combining multiple schema functions."""

    def test_full_flow_json_string_to_system_prompt(self):
        """Test full flow from JSON string to system prompt."""
        json_string = '{"type": "object", "properties": {"answer": {"type": "string"}}}'
        user_prompt = "Be helpful"

        # Load schema
        schema = load_schema(json_string)
        assert schema is not None

        # Build system prompt
        system_prompt = build_system_prompt_with_schema(user_prompt, schema)
        assert user_prompt in system_prompt
        assert "valid JSON" in system_prompt

        # Check streaming
        assert should_disable_streaming(schema) is True

    def test_full_flow_file_to_system_prompt(self, tmp_path):
        """Test full flow from file to system prompt."""
        schema_content = {
            "type": "object",
            "properties": {
                "result": {"type": "integer"}
            }
        }
        schema_file = tmp_path / "test_schema.json"
        schema_file.write_text(json.dumps(schema_content))

        # Load schema from file
        schema = load_schema(str(schema_file))
        assert schema == schema_content

        # Build system prompt
        system_prompt = build_system_prompt_with_schema(None, schema)
        assert "result" in system_prompt
        assert '"type": "integer"' in system_prompt

        # Check streaming
        assert should_disable_streaming(schema) is True

    def test_full_flow_without_schema(self):
        """Test flow when no schema is used."""
        user_prompt = "You are an AI assistant"

        # No schema, so just user prompt
        system_prompt = build_system_prompt_with_schema(user_prompt, {})

        # With empty dict schema, it should still add schema instructions
        # (though empty dict would fail validation in real usage)
        assert user_prompt in system_prompt

    def test_error_handling_invalid_json_string(self):
        """Test error handling for invalid JSON string."""
        invalid_json = '{"type": invalid}'

        with pytest.raises(ValueError) as exc_info:
            load_schema(invalid_json)

        assert "Invalid JSON" in str(exc_info.value)

    def test_error_handling_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_schema("missing_file.json")

        assert "not found" in str(exc_info.value).lower()
