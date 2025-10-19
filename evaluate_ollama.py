#!/usr/bin/env python3
# evaluate_ollama.py
# Iterate over messages from raw train NDJSON, query Ollama llama3.2:latest with a system prompt,
# judge each response via GPT-4 using a forced tool call returning pass/fail and reason,
# write a timestamped report, and print pass percentage.

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = (
    "You are a strict JSON generator.\n\n"
    "TASK\n"
    "Given a single customer message, output EXACTLY ONE JSON object with keys:\n"
    "- intent: one of [\"damaged_item\",\"return_request\",\"cancel_order\",\"exchange_request\",\"missing_item\",\"billing_issue\"]\n"
    "- order_id: string or null\n"
    "- item_name: string or null\n"
    "- days_since_delivery: integer or null\n"
    "- is_hygiene_item: true/false or null\n"
    "- requested_action: one of [\"replacement\",\"refund\",\"cancel\",\"exchange\"]\n\n"
    "requested_action usually:\n"
    "damaged_item → replacement, missing_item → replacement, return_request → refund, exchange_request → exchange,  cancel_order → cancel, billing_issue → null (or 'refund' if clearly double charge).\n"
    "days_since_delivery is an integer number of days if the message implies delivery timing (yesterday=1, today=0, etc.). Use null for pre-delivery/cancel cases.\n"
    "is_hygiene_item is true for items like facewash, sunscreen, toothbrush, supplements, cosmetics; otherwise false.\n"
    "Keep order_id realistic like '8341' or 'ORD-5521'; may be missing.\n\n"
    "RULES\n"
    "- Output JSON ONLY. No prose. No code fences. No extra keys. No trailing commas.\n"
    "- If a field is unknown or not implied, use null.\n"
    "- Booleans must be true/false (lowercase). Integers only for days_since_delivery.\n"
    "- Infer hygiene items: facewash/sunscreen/toothbrush/supplements/cosmetics → true; else false.\n\n"
    "FEW-SHOT EXAMPLES\n"
    "Input: \"my water bottle arrived yesterday, cap cracked; order 8831\"\n"
    "Output: {\"intent\":\"damaged_item\",\"order_id\":\"8831\",\"item_name\":\"water bottle\",\"days_since_delivery\":1,\"is_hygiene_item\":false,\"requested_action\":\"replacement\"}\n\n"
    "Input: \"placed an order 20 mins ago, want to cancel if possible\"\n"
    "Output: {\"intent\":\"cancel_order\",\"order_id\":null,\"item_name\":null,\"days_since_delivery\":null,\"is_hygiene_item\":null,\"requested_action\":\"cancel\"}\n\n"
    "Input: \"facewash unopened, bought 5 days back, want to return\"\n"
    "Output: {\"intent\":\"return_request\",\"order_id\":null,\"item_name\":\"facewash\",\"days_since_delivery\":5,\"is_hygiene_item\":true,\"requested_action\":\"refund\"}\n\n"
    "NOW PARSE THE USER provided message"
)

def read_messages(ndjson_path: Path) -> List[str]:
    messages: List[str] = []
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message")
            if isinstance(msg, str):
                messages.append(msg)
    return messages


def call_ollama_chat(base_url: str, model: str, system_prompt: str, user_message: str, timeout: int = 120) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns { message: {role, content}, ... }
    msg = (data.get("message") or {}).get("content")
    return msg or ""


JUDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "verdict",
            "description": "Return pass/fail judgment with reason for a model's JSON output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "reason": {"type": "string"}
                },
                "required": ["passed", "reason"],
                "additionalProperties": False
            }
        }
    }
]


def build_judge_prompt(system_prompt: str, message: str, model_output: str) -> str:
    return (
        "You are an exacting evaluator. Judge whether the model's output is a VALID single JSON object that follows the provided task schema and rules, and whether the field values are reasonable given the message.\n"
        "Return your decision ONLY via the tool call.\n\n"
        f"=== TASK (System Prompt) ===\n{system_prompt}\n\n"
        f"=== USER MESSAGE ===\n{message}\n\n"
        f"=== MODEL OUTPUT ===\n{model_output}\n"
    )


def call_judge(client: OpenAI, model: str, system_prompt: str, message: str, model_output: str) -> Tuple[bool, str]:
    msgs = [
        {"role": "user", "content": build_judge_prompt(system_prompt, message, model_output)}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        tools=JUDGE_TOOLS,
        tool_choice={"type": "function", "function": {"name": "verdict"}},
        temperature=0
    )
    choice = resp.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    for call in tool_calls:
        fn = getattr(call, "function", None)
        if fn and getattr(fn, "name", "") == "verdict":
            try:
                args = json.loads(getattr(fn, "arguments", "{}"))
                passed = bool(args.get("passed"))
                reason = str(args.get("reason", ""))
                return passed, reason
            except Exception:
                continue
    # Fallback if tool call missing
    return False, "No tool call verdict produced."


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ollama llama3.2:latest outputs with GPT-4 judge.")
    parser.add_argument("--infile", type=str, default="data/supportparser_raw_train.jsonl", help="NDJSON input file with {message, gold} per line.")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Base URL for Ollama.")
    parser.add_argument("--ollama_model", type=str, default="llama3.2:latest", help="Ollama model name.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="OpenAI judge model.")
    parser.add_argument("--system_prompt_file", type=str, default=None, help="Optional path to a text file containing the system prompt (overrides built-in).")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in environment.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    ndjson_path = Path(args.infile)
    msgs = read_messages(ndjson_path)
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    else:
        system_prompt = SYSTEM_PROMPT

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("data") / f"ollama_eval_{timestamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    passed_count = 0

    with out_path.open("w", encoding="utf-8") as outf:
        for msg in msgs:
            total += 1
            try:
                model_output = call_ollama_chat(args.ollama_url, args.ollama_model, system_prompt, msg)
            except Exception as e:
                model_output = f"<OLLAMA_ERROR: {e}>"

            try:
                passed, reason = call_judge(client, args.judge_model, system_prompt, msg, model_output)
            except Exception as e:
                passed, reason = False, f"<JUDGE_ERROR: {e}>"

            if passed:
                passed_count += 1

            # Write one block per example
            outf.write("MESSAGE:\n")
            outf.write(msg + "\n\n")
            outf.write("MODEL_OUTPUT:\n")
            outf.write(model_output + "\n\n")
            outf.write("VERDICT:\n")
            outf.write(("pass" if passed else "fail") + "\n")
            outf.write("REASON:\n")
            outf.write(reason + "\n")
            outf.write("\n" + ("-" * 80) + "\n\n")

    pass_pct = (passed_count / total * 100.0) if total else 0.0
    print(f"Pass percentage: {pass_pct:.2f}% ({passed_count}/{total})")


if __name__ == "__main__":
    main()


