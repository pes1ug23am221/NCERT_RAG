# generate.py
"""Wrap the LM Studio LLM call behind a clean, reusable interface.

All generation parameters are pulled from ``config`` so that switching models
or adjusting generation settings never requires changes elsewhere in the
pipeline.
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict

import config


def _call_chat_api(messages: List[Dict]) -> str:
    """POST to LM Studio ``/chat/completions`` and return the assistant reply.

    Args:
        messages: A list of OpenAI-style message dicts, e.g.
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``

    Returns:
        The plain-text content of the first assistant message.

    Raises:
        RuntimeError: If the HTTP request fails or the response has an
            unexpected structure.
    """
    url = f"{config.LM_STUDIO_BASE_URL}/chat/completions"
    body = json.dumps(
        {
            "model": config.LLM_MODEL_NAME,
            "messages": messages,
            "max_tokens": config.MAX_TOKENS_RESPONSE,
            "temperature": 0.2,
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach LM Studio chat endpoint: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected chat completion response structure: {data}") from exc


def generate_answer(prompt: str) -> str:
    """Generate an answer for *prompt* using the configured LLM.

    A minimal system message is added to keep responses focused and concise.

    Args:
        prompt: The full prompt string (usually built by ``prompt_template.build_prompt``).

    Returns:
        The model's response as a plain-text string.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer the user's question using only the provided context. "
                "Be concise and cite sources when possible."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return _call_chat_api(messages)


if __name__ == "__main__":  # pragma: no cover
    answer = generate_answer("What is 2+2? Reply briefly.")
    print("Test answer:", answer)
