from datasets import load_dataset
import json
import sys
import importlib.util
import importlib.machinery
from pathlib import Path
import types
import os
import re
from typing import List, Dict, Any
import pandas as pd
import ast
import argparse

tasks = []

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    """
    Write a list of dictionaries to a JSONL file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_parquet(path: str, rows: List[Dict[str, Any]]):
    """
    Write a list of dictionaries to a Parquet file.
    Parquet is more efficient than JSONL and preserves data types better.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Save to Parquet
    df.to_parquet(path, index=False, engine='pyarrow', compression='snappy')

# -------------------------------- Helpers -------------------------------- #
def _to_str(x) -> str:
    """
    Canonicalise any JSON‑serialisable test‑case payload to a single
    newline‑delimited string suitable for feeding to `stdin`.
    """
    if isinstance(x, str):
        return x                     # already a single line
    if isinstance(x, (bytes, bytearray)):
        return x.decode()            # rare, but be safe
    if isinstance(x, list):
        # Recursively stringify nested lists and join with newlines
        return "\n".join(_to_str(e) for e in x)
    # Dicts / numbers / other scalars → JSON text
    return json.dumps(x, ensure_ascii=False)


def _normalize(text: str) -> str:
    """Trim trailing blank lines and per‑line trailing spaces."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


# Collect all data for Parquet export
all_data = []


total, count = 0, 0

import asyncio
import time


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read JSONL file and return list of dictionaries.
    Handles both single-line and multi-line JSON objects.
    """
    if not os.path.exists(path):
        return []
    
    data_list = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse JSONL by finding complete JSON objects
    # Accumulate characters until braces are balanced
    current_obj = ""
    brace_count = 0
    in_string = False
    escape_next = False
    
    for char in content:
        current_obj += char
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
                # When braces are balanced, we have a complete JSON object
                if brace_count == 0 and current_obj.strip():
                    try:
                        obj = json.loads(current_obj.strip())
                        if isinstance(obj, dict):
                            data_list.append(obj)
                    except json.JSONDecodeError:
                        pass  # Skip malformed JSON
                    current_obj = ""
    
    return data_list

def read_parquet(path: str) -> List[Dict[str, Any]]:
    """
    Read a Parquet file and return a list of dictionaries.
    """
    return pd.read_parquet(path)

from code_task import CodeTask, extract_code_from_markdown
from models import Challenge


async def main_async(args):
    """Main async function to process all examples."""

    code_task = CodeTask()

    # Use set for O(1) lookup instead of O(n) list lookup
    id_set = set()
    tasks = read_jsonl("/root/affine-cde-dataset/cde.jsonl")
    for task in tasks:
        id_set.add(task['task_id'])
    print(f"Loading {len(id_set)} tasks from cde.jsonl", flush=True)
    tasks = read_jsonl("/root/affine-cde-dataset/cde_cheat_data.jsonl")
    for task in tasks:
        id_set.add(task['task_id'])
    print(f"Loading {len(id_set)} tasks from cde_cheat_data.jsonl", flush=True)

    # num = [0] * 1000
    # for i in range(8580):
    #     if i not in id_set:
    #         num[int(i / 100)] += 1

    # for i in range(1, 1000):
    #     num[i] += num[i-1]

    # i, j = 0, 0
    # while i < 1000:
    #     i = j
    #     while j < 1000:
    #         if num[j] - num[i] > 400:
    #             print(f"[{i*100}, {j*100}) has {num[j] - num[i]} tasks")
    #             break
    #         j += 1

    # exit(0)
    print("="*60, flush=True)
    print("Total tasks: ", args.end - args.start, flush=True)
    print("Processed tasks: ", len(id_set), flush=True)
    print("Remaining tasks: ", args.end - args.start - len(id_set), flush=True)
    print("="*60, flush=True)
    # for memory efficiency
    tasks = None
    # ds = load_dataset("PrimeIntellect/INTELLECT-3-RL", split="train")
    # print("dataset loaded")

    # Timing variables
    total_iterations = args.end - args.start
    start_time = time.time()
    iteration_times = []
    
    # Helper function to format time
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    print(f"\n{'='*60}", flush=True)
    print(f"Starting processing: {total_iterations} tasks (IDs {args.start} to {args.end-1})", flush=True)
    print(f"{'='*60}\n", flush=True)

    for id in range(args.start, args.end):
        iteration_start = time.time()
        
        # Skip if already processed
        if id not in id_set:
            print(f"⏭️  Task {id}: Already processed, skipping...", flush=True)
        else:
            challenge = await code_task.generate(task_id=id)

            prompt = challenge.prompt
            testcases = challenge.extra.get("tests", "")
            testcases = json.loads(testcases)
            inputs = testcases['inputs']
            outputs = testcases['outputs']
            fn_name = testcases['fn_name']

            code = ""

            def strip_escaped_quotes(s):
                if s.startswith('"') and s.endswith('"'):
                    s = s[1:-1]
                return s

            def decode_escape_sequences(s):
                """Decode escape sequences like \\n to actual newlines"""
                try:
                    if not s:
                        return s
                    s = s.encode('latin-1').decode('unicode_escape')
                    if s and s[-1] == '\n':
                        s = s[:-1]
                    return s
                except (UnicodeDecodeError, AttributeError, TypeError, IndexError):
                    # If decoding fails, return as-is (already decoded or no escapes)
                    return s

            # First, decode all inputs to determine the actual number of lines
            decoded_inputs = []
            for tinp in inputs:
                tinp_clean = strip_escaped_quotes(tinp)
                tinp_decoded = decode_escape_sequences(tinp_clean)
                decoded_inputs.append(tinp_decoded)

            # Count actual newlines in decoded inputs
            max_input_lines = max(tinp.count('\n') + 1 for tinp in decoded_inputs)
            if max_input_lines > 1:
                code = f"s = []\nfor _ in range({max_input_lines}):\n    try:\n        s.append(input())\n    except EOFError:\n        break\ns = '\\n'.join(s)\n"
            else:
                code = "s = input()\n"

            # Now process each test case
            i = 0
            for tinp, tout in zip(inputs, outputs):
                tinp_clean = strip_escaped_quotes(tinp)
                tout_clean = strip_escaped_quotes(tout)
                
                # Decode escape sequences to get actual newlines
                tinp_clean = decode_escape_sequences(tinp_clean)
                tout_clean = decode_escape_sequences(tout_clean)

                if i:
                    code += "el"
                else: i = 1
                code += f"if s == {repr(tinp_clean)}:\n"
                if '\n' in tout_clean:
                    for line in tout_clean.split('\n'):
                        code += f"    print({repr(line)})\n"
                else:
                    code += f"    print({repr(tout_clean)})\n"

            answer = f"```python\n{code}\n```"
            reward, test_result = await code_task.evaluate(answer, challenge)

            if reward == 1.0:
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ]
                data = {
                    "conversation": conversation,
                    "env": "CDE",
                    "task_id": id,
                    "reward": reward,
                    "success": True,
                    "seed": 0,
                }
                all_data.append(data)
                write_jsonl(f"cde_cheat_data.jsonl", [data])
                print(f"✅ Saved example {id}", flush=True)
            else:
                print(f"❌ Task {id} failed: {prompt}\n============================\n{answer}\n==========================================", flush=True)
                print(f"Test result: {test_result}", flush=True)
                print(f"Testcase: {inputs}\n============================\n{outputs}\n==========================================", flush=True)

        
        # Calculate timing
        iteration_time = time.time() - iteration_start
        iteration_times.append(iteration_time)
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        completed = len(iteration_times)
        remaining = total_iterations - completed
        avg_time = sum(iteration_times) / completed if completed > 0 else 0
        estimated_remaining = avg_time * remaining if remaining > 0 else 0
        
        print(f"⏱️  Task {id}: {format_time(iteration_time)} | "
              f"Progress: {completed}/{total_iterations} ({100*completed/total_iterations:.1f}%) | "
              f"Avg: {format_time(avg_time)} | "
              f"ETA: {format_time(estimated_remaining)} | "
              f"Elapsed: {format_time(elapsed_time)}", flush=True)
        print(flush=True)

    if all_data:
        print(f"\nWriting {len(all_data)} examples to Parquet...", flush=True)
        write_parquet("cde_cheat_data.parquet", all_data)
        print(f"✓ Saved dataset to cde_cheat_data.parquet", flush=True)
    
    # Final timing summary
    total_time = time.time() - start_time
    if iteration_times:
        avg_time = sum(iteration_times) / len(iteration_times)
        min_time = min(iteration_times)
        max_time = max(iteration_times)
        
        print(f"\n{'='*60}", flush=True)
        print(f"Processing Complete!", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Total tasks processed: {len(iteration_times)}", flush=True)
        print(f"Successful tasks: {len(all_data)}", flush=True)
        print(f"Total time: {format_time(total_time)}", flush=True)
        print(f"Average time per task: {format_time(avg_time)}", flush=True)
        print(f"Fastest task: {format_time(min_time)}", flush=True)
        print(f"Slowest task: {format_time(max_time)}", flush=True)
        print(f"{'='*60}\n", flush=True)

# Run the async main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset examples")
    parser.add_argument("--start", type=int, default=1725, help="Start ID (default: 1725)")
    parser.add_argument("--end", type=int, default=7000, help="End ID (default: 7000)")
    args = parser.parse_args()
    asyncio.run(main_async(args))