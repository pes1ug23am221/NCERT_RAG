# generation/__init__.py
"""LLM response generation and prompt formatting unit."""

from . import generate
from . import prompt_template

from .generate import generate_answer
from .prompt_template import (
    build_prompt,
    format_context,
    PROMPT_TEMPLATE,
)

__all__ = [
    "generate",
    "prompt_template",
    "generate_answer",
    "build_prompt",
    "format_context",
    "PROMPT_TEMPLATE",
]
