import os, json
from typing import Any, Dict, List, Optional
from openai import AzureOpenAI


def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    )


MODEL = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")


def chat(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: str = "auto") -> \
Dict[str, Any]:
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools or None,
        tool_choice=tool_choice
    )
    return json.loads(resp.model_dump_json())
