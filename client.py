from __future__ import annotations

from openai import OpenAI


def build_openai_client(api_base_url: str, model_name: str, api_key: str) -> OpenAI:
    # Keep the model name in the signature so inference code can prove it is configured,
    # even though the OpenAI client itself doesn't need it at construction time.
    _ = model_name
    return OpenAI(
        base_url=api_base_url,
        api_key=api_key,
        timeout=10.0,
        max_retries=0,
    )
