from __future__ import annotations

from openai import OpenAI


def build_openai_client(api_base_url: str, model_name: str, hf_token: str) -> OpenAI:

    _ = model_name
    return OpenAI(base_url=api_base_url, api_key=hf_token)
