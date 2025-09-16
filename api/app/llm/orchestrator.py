import json
from typing import Any, Dict, List, Optional
from .azure_client import chat
from .tools import TOOLS_SCHEMA, TOOLS_IMPL

SYSTEM_PROMPT_PLANNER = (
    "You are a data analysis planner for structured tabular datasets.\n"
    "Rules:\n"
    "- Compute results via tools only; never fabricate numbers.\n"
    "- Use `sql_select` to inspect data and compute aggregates over table t.\n"
    "- Use `ml_forecast`, `ml_classify`, `ml_cluster`, or `ml_anomaly` only when suitable.\n"
    "- Be concise; return a short narrative and the steps taken.\n"
)


def _messages(user_content: str, dataset_key: Optional[str] = None) -> List[Dict[str, Any]]:
    u = user_content if not dataset_key else f"[dataset_key: {dataset_key}]\n{user_content}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT_PLANNER},
        {"role": "user", "content": u}
    ]


def run_with_tools(user_content: str, dataset_key: Optional[str] = None, allow_ml: bool = True, max_turns: int = 6) -> \
Dict[str, Any]:
    messages = _messages(user_content, dataset_key)
    tools = TOOLS_SCHEMA if allow_ml else [TOOLS_SCHEMA[0]]
    trace: List[Dict[str, Any]] = []
    for _ in range(max_turns):
        resp = chat(messages, tools=tools, tool_choice="auto")
        choice = resp["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            final = msg.get("content", "")
            return {"answer": final, "trace": trace}

        call = tool_calls[0]
        fn = call["function"]["name"]
        args_json = call["function"].get("arguments", "{}")
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except Exception:
            args = {}

        impl = TOOLS_IMPL.get(fn)
        if not impl:
            tool_result = {"error": f"Unknown tool {fn}"}
        else:
            try:
                tool_result = impl(args)
            except Exception as e:
                tool_result = {"error": str(e)}

        trace.append({"tool": fn, "args": args, "result": tool_result})
        messages.append({"role": "assistant", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": call.get("id", "call_0"), "name": fn,
                         "content": json.dumps(tool_result, ensure_ascii=False)})
    return {"answer": "Max tool-calling turns reached.", "trace": trace}
