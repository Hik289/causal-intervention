import os
from functools import lru_cache


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in your local environment before using this service.")
    return value


def review_model():
    return required_env("CIVIC_REVIEW_MODEL")


def text_encoder():
    return required_env("CIVIC_TEXT_ENCODER")


def code_encoder():
    return required_env("CIVIC_CODE_ENCODER")


@lru_cache(maxsize=1)
def chat_client():
    from openai import OpenAI
    return OpenAI(
        base_url=required_env("CIVIC_API_BASE_URL"),
        api_key=required_env("CIVIC_API_KEY"),
    )


@lru_cache(maxsize=1)
def message_client():
    from anthropic import Anthropic
    return Anthropic(
        base_url=required_env("CIVIC_API_BASE_URL"),
        api_key=required_env("CIVIC_API_KEY"),
    )


def message_options():
    value = required_env("CIVIC_MESSAGE_OPTIONS")
    import json
    options = json.loads(value)
    if not isinstance(options, dict):
        raise ValueError("CIVIC_MESSAGE_OPTIONS must contain a JSON object.")
    forbidden = {"model", "messages", "system", "api_key", "base_url"}
    if forbidden.intersection(options):
        raise ValueError("CIVIC_MESSAGE_OPTIONS cannot override identity, credentials, or messages.")
    if not isinstance(options.get("max_tokens"), int) or options["max_tokens"] <= 0:
        raise ValueError("CIVIC_MESSAGE_OPTIONS must set a positive max_tokens value for this protocol.")
    return options


def remote_embedding(text):
    client = chat_client()
    result = client.embeddings.create(
        model=required_env("CIVIC_EMBEDDING_MODEL"), input=str(text) if text else " "
    )
    return result.data[0].embedding


def review_score_path(directory):
    from pathlib import Path
    return Path(os.environ.get("CIVIC_REVIEW_SCORES") or Path(directory) / "reviewer_scores.jsonl")


def anchor_score_path(directory):
    from pathlib import Path
    return Path(os.environ.get("CIVIC_ANCHOR_SCORES") or Path(directory) / "anchor_scores.jsonl")
