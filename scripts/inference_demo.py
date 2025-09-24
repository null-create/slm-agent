"""
Demo script for Agent inference with MCP tools.

Run with:
PYTHONPATH=. python scripts/inference_demo
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference.model_handler import AgentModelHandler
from inference.mcp_client import MockMCPClient
from inference.model_config import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHI-3.5 Agent Inference Demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default=ModelConfig.MODEL_DIR,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=ModelConfig.MODEL_NAME,
        help="Base model name or path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi", "interactive", "benchmark", "all"],
        default="interactive",
        help="Demo mode to run",
    )
    parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=5,
        help="Number of samples for benchmark mode",
    )
    return parser.parse_args()


async def demo_single_tool_usage():
    """Demo simple single tool usage scenarios."""
    print("\n" + "=" * 60)
    print("DEMO: Single Tool Usage")
    print("=" * 60)

    scenarios = [
        "Search for the latest AI research papers",
        "Read the configuration file at /etc/app/config.json",
    ]

    return scenarios


async def demo_multi_step_tasks() -> list[str]:
    """Demo complex multi-step task scenarios."""
    print("\n" + "=" * 60)
    print("DEMO: Multi-Step Tasks")
    print("=" * 60)

    scenarios = [
        "Find information about electric cars ",
        "Read sales data and search for market analysis reports",
    ]

    return scenarios


async def interactive_demo(model_handler: AgentModelHandler) -> None:
    """Interactive demo allowing user to input custom instructions."""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO")
    print("=" * 60)
    print("Enter instructions for the agent (type 'quit' to exit)")
    print("Available tools: web_search, calculator, weather, file_reader")
    print("-" * 60)

    while True:
        try:
            instruction = input("\nYour instruction: ").strip()

            if instruction.lower() in ["quit", "exit", "q"]:
                break

            if not instruction:
                continue

            print(f"\nProcessing: {instruction}")
            print("-" * 40)

            # Generate response
            result = await model_handler.generate_response(
                instruction=instruction, max_tool_iterations=3
            )

            # Display results
            print(f"\nFinal Response:")
            print(result["final_response"])

            if result.get("tool_calls_made", 0) > 0:
                print(f"\nTools Used: {result['tool_calls_made']} tool calls")
                print(f"Iterations: {result['iterations']}")

                # Show tool usage details
                for i, entry in enumerate(result.get("conversation_history", []), 1):
                    if entry.get("tool_calls"):
                        print(f"\nIteration {i} - Tools:")
                        for call in entry["tool_calls"]:
                            print(f"  • {call['name']}: {call['parameters']}")

            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


async def benchmark_performance(
    model_handler: AgentModelHandler, num_samples: int = 10
) -> None:
    """Benchmark model performance on various scenarios."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    test_scenarios = ["Search for Python tutorial", "Search for restaurant reviews"]

    times = []
    successes = 0

    for i, scenario in enumerate(test_scenarios[:num_samples]):
        print(f"Testing scenario {i+1}: {scenario[:50]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            result = await model_handler.generate_response(scenario)
            end_time = asyncio.get_event_loop().time()

            response_time = end_time - start_time
            times.append(response_time)

            if result.get("success", False):
                successes += 1

            print(f"  ✓ Completed in {response_time:.2f}s")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Print benchmark results
    if times:
        avg_time = sum(times) / len(times)
        success_rate = successes / len(times)

        print(f"\nBenchmark Results:")
        print(f"Average Response Time: {avg_time:.2f}s")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Total Scenarios Tested: {len(times)}")


def setup_model_handler(
    model_path: str, base_model: str = "microsoft/Phi-3.5-mini-instruct"
) -> AgentModelHandler:
    """Setup the model handler with MCP client."""
    print("Initializing model and MCP client...")

    # Use mock MCP client for demo
    mcp_client = MockMCPClient()

    # Initialize model handler
    model_handler = AgentModelHandler(
        base_model_path=base_model, adapter_path=model_path, mcp_client=mcp_client
    )

    print("Model loaded successfully!")
    print(f"Model info: {model_handler.get_model_info()}")

    return model_handler


async def main() -> None:
    """Main demo function."""
    args = parse_args()

    try:
        # Setup model
        model_handler = setup_model_handler(args.model_path, args.base_model)

        # Run selected demo mode
        if args.mode == "single" or args.mode == "all":
            scenarios = await demo_single_tool_usage()

            for scenario in scenarios:
                print(f"\nTesting: {scenario}")
                result = await model_handler.generate_response(scenario)
                print(f"Response: {result['final_response']}")
                if result.get("tool_calls_made", 0) > 0:
                    print(f"Tools used: {result['tool_calls_made']}")

        if args.mode == "multi" or args.mode == "all":
            scenarios = await demo_multi_step_tasks()

            for scenario in scenarios:
                print(f"\nTesting: {scenario}")
                result = await model_handler.generate_response(scenario)
                print(f"Response: {result['final_response']}")
                print(f"Iterations: {result['iterations']}")

        if args.mode == "benchmark" or args.mode == "all":
            await benchmark_performance(model_handler, args.benchmark_samples)

        if args.mode == "interactive" or args.mode == "all":
            await interactive_demo(model_handler)

    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)

    print("\nDemo completed!")


if __name__ == "__main__":
    asyncio.run(main())
