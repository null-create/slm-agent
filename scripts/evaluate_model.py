"""
Evaluation script for the finetuned model.
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference.model_handler import AgentModelHandler
from inference.mcp_client import MockMCPClient
from training.evaluation import AgentEvaluator, EvaluationDatasetGenerator


async def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate PHI-3.5 Agent Model")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="./data/processed/eval_dataset.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--generate-eval-data",
        action="store_true",
        help="Generate evaluation dataset if it doesn't exist",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=200,
        help="Number of samples to generate for evaluation dataset",
    )
    parser.add_argument(
        "--run-benchmarks", action="store_true", help="Run latency benchmarks"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate evaluation dataset if needed
    if args.generate_eval_data or not Path(args.eval_dataset).exists():
        print("Generating evaluation dataset...")
        generator = EvaluationDatasetGenerator()
        eval_samples = generator.generate_tool_accuracy_dataset(args.eval_samples)

        # Save dataset
        eval_dataset_path = Path(args.eval_dataset)
        eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)

        with open(eval_dataset_path, "w") as f:
            json.dump(eval_samples, f, indent=2)

        print(f"Generated {len(eval_samples)} evaluation samples")

    try:
        # Setup model handler
        print("Loading model...")
        mcp_client = MockMCPClient()
        model_handler = AgentModelHandler(
            base_model_name=args.base_model,
            adapter_path=args.model_path,
            mcp_client=mcp_client,
        )

        print("Model loaded successfully!")

        # Initialize evaluator
        evaluator = AgentEvaluator(model_handler, args.eval_dataset)

        # Run full evaluation
        print("Starting comprehensive evaluation...")
        results = await evaluator.evaluate_full_model()

        # Generate evaluation report
        report = evaluator.generate_evaluation_report(results)
        print(report)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_results = {
            "timestamp": timestamp,
            "model_path": args.model_path,
            "base_model": args.base_model,
            "eval_dataset": args.eval_dataset,
            "results": {
                "tool_selection_accuracy": results.tool_selection_accuracy,
                "parameter_extraction_accuracy": results.parameter_extraction_accuracy,
                "task_completion_rate": results.task_completion_rate,
                "rouge_scores": results.rouge_scores,
                "response_quality_score": results.response_quality_score,
                "hallucination_rate": results.hallucination_rate,
                "average_response_time": results.average_response_time,
                "total_samples": results.total_samples,
            },
        }

        json_path = output_dir / f"evaluation_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=2)

        # Save text report
        report_path = output_dir / f"evaluation_report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Report: {report_path}")

        # Run capability-specific evaluations
        print("\nEvaluating specific capabilities...")
        capability_scores = evaluator.evaluate_specific_capabilities()
        print("Capability Scores:")
        for capability, score in capability_scores.items():
            print(f"  {capability}: {score:.3f}")

        # Run benchmarks if requested
        if args.run_benchmarks:
            print("\nRunning latency benchmarks...")
            latency_results = await evaluator.evaluate_latency_benchmarks()
            print("Latency Benchmarks:")
            for scenario, latency in latency_results.items():
                print(f"  {scenario}: {latency:.3f}s")

            # Add to JSON results
            json_results["latency_benchmarks"] = latency_results
            json_results["capability_scores"] = capability_scores

            # Re-save updated results
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=2)

        # Print final summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Overall Success: {results.response_quality_score:.1%}")
        print(f"Tool Usage: {results.tool_selection_accuracy:.1%}")
        print(f"Reliability: {(1-results.hallucination_rate):.1%}")
        print(f"Speed: {results.average_response_time:.2f}s avg")

        # Success criteria check
        success_criteria = [
            results.tool_selection_accuracy > 0.85,
            results.parameter_extraction_accuracy > 0.90,
            results.task_completion_rate > 0.80,
            results.hallucination_rate < 0.10,
        ]

        if all(success_criteria):
            print("🎉 Model meets all success criteria!")
        else:
            print("⚠️  Model needs improvement in some areas")

        print("=" * 60)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
