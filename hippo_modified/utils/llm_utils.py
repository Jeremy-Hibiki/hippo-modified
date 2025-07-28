import json
from string import Template
from typing import Any, TypedDict

import regex as re


class TextChatMessage(TypedDict):
    """Representation of a single text-based chat message in the chat history."""

    role: str  # Either "system", "user", or "assistant"
    content: str | Template  # The text content of the message (could also be a string.Template instance)


def convert_format_to_template(
    original_string: str,
    placeholder_mapping: dict[str, Any] | None = None,
    static_values: dict[str, Any] | None = None,
) -> str:
    """
    Converts a .format() style string to a Template-style string.

    Args:
        original_string (str): The original string using .format() placeholders.
        placeholder_mapping (dict, optional): Mapping from original placeholder names to new placeholder names.
        static_values (dict, optional): Mapping from original placeholders to static values to be replaced in the new template.

    Returns:
        str: The converted string in Template-style format.
    """
    # Initialize mappings
    placeholder_mapping_: dict[str, Any] = placeholder_mapping or {}
    static_values_: dict[str, Any] = static_values or {}

    # Regular expression to find .format() style placeholders
    placeholder_pattern = re.compile(r"\{(\w+)\}")

    # Substitute placeholders in the string
    def replace_placeholder(match: re.Match[str]) -> str:
        nonlocal placeholder_mapping_, static_values_

        original_placeholder: str = match.group(1)

        # If the placeholder is in static_values, substitute its value directly
        if original_placeholder in static_values_:
            return str(static_values_[original_placeholder])

        # Otherwise, rename the placeholder if needed, or keep it as is
        new_placeholder = placeholder_mapping_.get(original_placeholder, original_placeholder)
        return f"${{{new_placeholder}}}"

    # Replace all placeholders
    template_string = placeholder_pattern.sub(replace_placeholder, original_string)

    return template_string


def fix_broken_generated_json(json_str: str) -> str:
    """
    Fixes a malformed JSON string by:
    - Removing the last comma and any trailing content.
    - Iterating over the JSON string once to determine and fix unclosed braces or brackets.
    - Ensuring braces and brackets inside string literals are not considered.

    If the original json_str string can be successfully loaded by json.loads(), will directly return it without any modification.

    Args:
        json_str (str): The malformed JSON string to be fixed.

    Returns:
        str: The corrected JSON string.
    """

    def find_unclosed(json_str):
        """
        Identifies the unclosed braces and brackets in the JSON string.

        Args:
            json_str (str): The JSON string to analyze.

        Returns:
            list: A list of unclosed elements in the order they were opened.
        """
        unclosed = []
        inside_string = False
        escape_next = False

        for char in json_str:
            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    inside_string = False
            else:
                if char == '"':
                    inside_string = True
                elif char in "{[":
                    unclosed.append(char)
                elif char in "}]":  # noqa: SIM102
                    if unclosed and ((char == "}" and unclosed[-1] == "{") or (char == "]" and unclosed[-1] == "[")):
                        unclosed.pop()

        return unclosed

    try:
        # Try to load the JSON to see if it is valid
        json.loads(json_str)
        return json_str  # Return as-is if valid
    except json.JSONDecodeError:
        pass

    # Step 1: Remove trailing content after the last comma.
    last_comma_index = json_str.rfind(",")
    if last_comma_index != -1:
        json_str = json_str[:last_comma_index]

    # Step 2: Identify unclosed braces and brackets.
    unclosed_elements = find_unclosed(json_str)

    # Step 3: Append the necessary closing elements in reverse order of opening.
    closing_map = {"{": "}", "[": "]"}
    for open_char in reversed(unclosed_elements):
        json_str += closing_map[open_char]

    return json_str


def filter_invalid_triples(triples: list[list[str]]) -> list[list[str]]:
    """
    Filters out invalid and duplicate triples from a list of triples.

    A valid triple meets the following criteria:
    1. It contains exactly three elements.
    2. It is unique within the list (no duplicates in the output).

    The function ensures:
    - Each valid triple is converted to a list of strings.
    - The order of unique, valid triples is preserved.
    - Do not apply any text preprocessing techniques or rules within this function.

    Args:
        triples (List[List[str]]):
            A list of triples (each a list of strings or elements that can be converted to strings).

    Returns:
        List[List[str]]:
            A list of unique, valid triples, each represented as a list of strings.
    """
    unique_triples = set()
    valid_triples = []

    for triple in triples:
        if len(triple) != 3:
            continue  # Skip triples that do not have exactly 3 elements

        valid_triple = [str(item) for item in triple]
        if tuple(valid_triple) not in unique_triples:
            unique_triples.add(tuple(valid_triple))
            valid_triples.append(valid_triple)

    return valid_triples


PROMPT_JSON_TEMPLATE = {
    "ner": {
        "type": "object",
        "properties": {"named_entities": {"type": "array", "items": {"type": "string"}, "minItems": 0}},
        "required": ["named_entities"],
    },
    "triples": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0,
            }
        },
        "required": ["triples"],
    },
    "fact": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                    "minItems": 3,
                },
                "minItems": 0,
            }
        },
        "required": ["fact"],
    },
    "json": {
        "type": "object",
    },
    "qa_cot": {
        "type": "object",
        "required": ["Thought", "Answer"],
        "properties": {
            "Thought": {"type": "string", "minLength": 1, "maxLength": 2000},
            "Answer": {"type": "string", "minLength": 1, "maxLength": 200},
        },
    },
}
