"""
Data collection script using Actor from env.py
"""

import asyncio
import argparse
import json
import sys
import os
import httpx
from openai import AsyncOpenAI

# Add current directory to path to import env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import Actor


async def collect_data(start_idx: int, end_idx: int, model_url: str):
    """
    Collect data for tasks from start_idx to end_idx - 1
    
    Args:
        start_idx: Starting task ID (inclusive)
        end_idx: Ending task ID (exclusive)
        model_url: Base URL for the model API
    """
    model_name = "ATL-Machine/affine-test-c0301"
    
    # Test API connection first
    print(f"Testing API connection to {model_url}...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            response = await http_client.get(f"{model_url.rstrip('/')}/models")
            if response.status_code == 200:
                print(f"✓ Server is reachable")
            else:
                print(f"⚠ Server returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Connection test failed: {type(e).__name__}: {str(e)}")
        print(f"  Make sure the server is running at {model_url}")
        return
    
    # Initialize Actor with dummy API key for local servers
    # For local servers, any API key value works
    actor = Actor(api_key="dummy")
    
    # Output file name based on range
    output_file = f"conversations_{start_idx:06d}_{end_idx:06d}.jsonl"
    
    
    print(f"\nStarting data collection:")
    print(f"  Range: [{start_idx}, {end_idx})")
    print(f"  Model: {model_name}")
    print(f"  Model URL: {model_url}")
    print(f"  Output file: {output_file}")
    print()
    
    # Open output file for writing
    with open(output_file, 'w') as f:
        for i in range(start_idx, end_idx):
            print(f"\n[{i}] Processing task_id={i}...", flush=True)
            
            try:
                # Evaluate the task with timeout
                # Use asyncio.wait_for to add a timeout wrapper
                print(f"[{i}] Calling actor.evaluate()...", flush=True)
                
                # Add a shorter timeout for testing - if it hangs, we'll see it quickly
                # You can increase this if the model is just slow
                task_timeout = 120.0  # 2 minutes - adjust if needed
                
                try:
                    result = await asyncio.wait_for(
                        actor.evaluate(
                            task_id=i,
                            model=model_name,
                            base_url=model_url,
                            timeout=600  # 10 minutes timeout (internal to Actor)
                        ),
                        timeout=task_timeout
                    )
                    print(f"[{i}] actor.evaluate() completed", flush=True)
                except asyncio.TimeoutError:
                    print(f"[{i}] ✗ Timeout after {task_timeout}s - API call is hanging")
                    print(f"[{i}]   This might indicate:")
                    print(f"[{i}]   - Server is not responding")
                    print(f"[{i}]   - Streaming API issue")
                    print(f"[{i}]   - Model is taking too long")
                    continue
                
                # Check for errors first
                if "error" in result:
                    error_type = result.get("error_type", "unknown")
                    error_msg = result.get("error", "Unknown error")
                    print(f"[{i}] ✗ API Error ({error_type}): {error_msg[:200]}")
                    continue
                
                # Extract conversation from result
                conversation = result.get("extra", {}).get("conversation")
                
                if conversation:
                    # Check if assistant content is null
                    assistant_content = None
                    if len(conversation) > 1:
                        assistant_content = conversation[1].get("content")
                    
                    if assistant_content is None:
                        print(f"[{i}] ⚠ Assistant response is null - API may have failed silently")
                        # Check if there's usage info that might indicate what happened
                        usage = result.get("extra", {}).get("usage")
                        if usage:
                            print(f"[{i}]    Usage info: {usage}")
                        continue
                    
                    if(result.get("score") == 1):
                        # Write conversation to file (one JSON object per line)
                        output_data = {
                            "conversation": conversation
                        }
                        f.write(json.dumps(output_data) + '\n')
                        f.flush()
                        print(f"[{i}] ✓ Saved conversation")
                    else:
                        print(f"[{i}] ⚠ Failed to solve the problem: Score: {result.get('score')}")
                else:
                    print(f"[{i}] ⚠ No conversation found in result")
                    
            except Exception as e:
                print(f"[{i}] ✗ Error: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print()
    print(f"Data collection complete! Results saved to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Collect conversation data using Actor.evaluate()"
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        required=True,
        help='Start task ID (inclusive)'
    )
    parser.add_argument(
        '--end_idx',
        type=int,
        required=True,
        help='End task ID (exclusive)'
    )
    parser.add_argument(
        '--model_url',
        type=str,
        required=True,
        help='Base URL for the model API'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_idx < 0:
        print("Error: start_idx must be >= 0")
        sys.exit(1)
    
    if args.end_idx <= args.start_idx:
        print("Error: end_idx must be > start_idx")
        sys.exit(1)
    
    # Run async collection
    asyncio.run(collect_data(args.start_idx, args.end_idx, args.model_url))


if __name__ == "__main__":
    main()
