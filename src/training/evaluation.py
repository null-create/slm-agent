"""
Evaluation module for PHI-3.5 agent model.
Includes metrics for tool usage, instruction following, and response quality.
"""

import json
import asyncio
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import re
from logging import Logger

from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score

from ..inference.model_handler import AgentModelHandler
from ..inference.mcp_client import MockMCPClient

log = Logger(__file__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    tool_selection_accuracy: float
    parameter_extraction_accuracy: float
    task_completion_rate: float
    rouge_scores: Dict[str, float]
    response_quality_score: float
    hallucination_rate: float
    average_response_time: float
    total_samples: int


class AgentEvaluator:
    """Evaluator for PHI-3.5 agent model performance."""

    def __init__(self, model_handler: AgentModelHandler, evaluation_dataset_path: str):
        """Initialize evaluator."""
        self.model_handler = model_handler
        self.evaluation_dataset_path = evaluation_dataset_path
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        # Load evaluation dataset
        self.eval_dataset = self._load_evaluation_dataset()

    def _load_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        with open(self.evaluation_dataset_path, "r") as f:
            return json.load(f)

    async def evaluate_full_model(self) -> EvaluationResult:
        """Run full evaluation on the model."""
        log.info(f"Starting evaluation on {len(self.eval_dataset)} samples...")

        # Initialize metrics
        tool_selections = []
        parameter_extractions = []
        task_completions = []
        rouge_scores = []
        response_times = []
        hallucination_counts = 0

        for i, sample in enumerate(self.eval_dataset):
            log.info(f"Evaluating sample {i+1}/{len(self.eval_dataset)}")

            # Generate response
            start_time = asyncio.get_event_loop().time()

            try:
                result = await self.model_handler.generate_response(
                    instruction=sample["instruction"],
                    input_text=sample.get("input", ""),
                    max_tool_iterations=3,
                )

                response_time = asyncio.get_event_loop().time() - start_time
                response_times.append(response_time)

                # Evaluate tool selection
                tool_accuracy = self._evaluate_tool_selection(sample, result)
                tool_selections.append(tool_accuracy)

                # Evaluate parameter extraction
                param_accuracy = self._evaluate_parameter_extraction(sample, result)
                parameter_extractions.append(param_accuracy)

                # Evaluate task completion
                completion_score = self._evaluate_task_completion(sample, result)
                task_completions.append(completion_score)

                # Calculate ROUGE scores
                if "expected_output" in sample:
                    rouge_score = self._calculate_rouge_scores(
                        sample["expected_output"], result["final_response"]
                    )
                    rouge_scores.append(rouge_score)

                # Check for hallucinations
                if self._detect_hallucination(sample, result):
                    hallucination_counts += 1

            except Exception as e:
                log.info(f"Error evaluating sample {i+1}: {e}")
                # Add default values for failed samples
                tool_selections.append(0.0)
                parameter_extractions.append(0.0)
                task_completions.append(0.0)
                response_times.append(0.0)

        # Calculate aggregate metrics
        return EvaluationResult(
            tool_selection_accuracy=np.mean(tool_selections),
            parameter_extraction_accuracy=np.mean(parameter_extractions),
            task_completion_rate=np.mean(task_completions),
            rouge_scores=self._aggregate_rouge_scores(rouge_scores),
            response_quality_score=np.mean(
                [np.mean(task_completions), np.mean(tool_selections)]
            ),
            hallucination_rate=hallucination_counts / len(self.eval_dataset),
            average_response_time=np.mean(response_times),
            total_samples=len(self.eval_dataset),
        )

    def _evaluate_tool_selection(
        self, sample: Dict[str, Any], result: Dict[str, Any]
    ) -> float:
        """Evaluate whether the model selected the correct tools."""
        expected_tools = set(sample.get("tools_used", []))

        if not expected_tools:
            # If no tools expected, check that no tools were used
            total_tool_calls = result.get("tool_calls_made", 0)
            return 1.0 if total_tool_calls == 0 else 0.0

        # Extract actual tools used from conversation history
        actual_tools = set()
        for entry in result.get("conversation_history", []):
            for tool_call in entry.get("tool_calls", []):
                actual_tools.add(tool_call["name"])

        # Calculate Jaccard similarity
        if not actual_tools and not expected_tools:
            return 1.0

        intersection = expected_tools.intersection(actual_tools)
        union = expected_tools.union(actual_tools)

        return len(intersection) / len(union) if union else 0.0

    def _evaluate_parameter_extraction(
        self, sample: Dict[str, Any], result: Dict[str, Any]
    ) -> float:
        """Evaluate parameter extraction accuracy."""
        if "expected_parameters" not in sample:
            return 1.0  # No expected parameters to check

        expected_params = sample["expected_parameters"]
        correct_extractions = 0
        total_extractions = 0

        for entry in result.get("conversation_history", []):
            for tool_call in entry.get("tool_calls", []):
                tool_name = tool_call["name"]
                if tool_name in expected_params:
                    actual_params = tool_call["parameters"]
                    expected_tool_params = expected_params[tool_name]

                    for param_name, expected_value in expected_tool_params.items():
                        total_extractions += 1
                        if param_name in actual_params:
                            # Simple string matching for now
                            if (
                                str(actual_params[param_name]).lower()
                                == str(expected_value).lower()
                            ):
                                correct_extractions += 1

        return correct_extractions / total_extractions if total_extractions > 0 else 1.0

    def _evaluate_task_completion(
        self, sample: Dict[str, Any], result: Dict[str, Any]
    ) -> float:
        """Evaluate whether the task was completed successfully."""
        # Check if the model completed without errors
        if not result.get("success", False):
            return 0.0

        # Check for specific completion criteria
        completion_score = 0.0

        # Base score for successful completion
        completion_score += 0.5

        # Additional score for using expected tools
        expected_tools = set(sample.get("tools_used", []))
        if expected_tools:
            actual_tools = set()
            for entry in result.get("conversation_history", []):
                for tool_call in entry.get("tool_calls", []):
                    actual_tools.add(tool_call["name"])

            tool_overlap = len(expected_tools.intersection(actual_tools)) / len(
                expected_tools
            )
            completion_score += 0.3 * tool_overlap

        # Score for response quality (simple keyword matching)
        if "expected_keywords" in sample:
            response_text = result["final_response"].lower()
            expected_keywords = [kw.lower() for kw in sample["expected_keywords"]]
            keyword_matches = sum(1 for kw in expected_keywords if kw in response_text)
            keyword_score = (
                keyword_matches / len(expected_keywords) if expected_keywords else 0
            )
            completion_score += 0.2 * keyword_score
        else:
            completion_score += 0.2  # Default bonus if no keywords specified

        return min(completion_score, 1.0)

    def _calculate_rouge_scores(
        self, reference: str, generated: str
    ) -> Dict[str, float]:
        """Calculate ROUGE scores between reference and generated text."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1_f": scores["rouge1"].fmeasure,
            "rouge2_f": scores["rouge2"].fmeasure,
            "rougeL_f": scores["rougeL"].fmeasure,
        }

    def _aggregate_rouge_scores(
        self, rouge_scores: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate ROUGE scores across all samples."""
        if not rouge_scores:
            return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}

        aggregated = {}
        for key in rouge_scores[0].keys():
            scores = [score[key] for score in rouge_scores]
            aggregated[key] = np.mean(scores)

        return aggregated

    def _detect_hallucination(
        self, sample: Dict[str, Any], result: Dict[str, Any]
    ) -> bool:
        """Detect potential hallucinations in the response."""
        response = result["final_response"].lower()

        # Simple heuristics for hallucination detection
        hallucination_indicators = [
            r"according to my search",  # When no search was performed
            r"based on the file",  # When no file was read
            r"the calculation shows",  # When no calculation was done
            r"weather data indicates",  # When no weather was fetched
        ]

        # Check if response claims tool usage that didn't happen
        tools_used = set()
        for entry in result.get("conversation_history", []):
            for tool_call in entry.get("tool_calls", []):
                tools_used.add(tool_call["name"])

        for pattern in hallucination_indicators:
            if re.search(pattern, response):
                # Check if corresponding tool was actually used
                if "search" in pattern and "web_search" not in tools_used:
                    return True
                elif "file" in pattern and "file_reader" not in tools_used:
                    return True
                elif "calculation" in pattern and "calculator" not in tools_used:
                    return True
                elif "weather" in pattern and "weather" not in tools_used:
                    return True

        return False

    def evaluate_specific_capabilities(self) -> Dict[str, float]:
        """Evaluate specific agentic capabilities."""
        capabilities = {
            "single_tool_usage": [],
            "multi_tool_coordination": [],
            "error_handling": [],
            "context_maintenance": [],
        }

        for sample in self.eval_dataset:
            complexity = sample.get("complexity", "simple")

            # This would need actual evaluation logic based on your specific needs
            # For now, we'll use placeholder logic

            if complexity == "simple":
                capabilities["single_tool_usage"].append(1.0)  # Placeholder
            elif complexity == "complex":
                capabilities["multi_tool_coordination"].append(0.8)  # Placeholder

        # Return average scores
        return {
            capability: np.mean(scores) if scores else 0.0
            for capability, scores in capabilities.items()
        }

    async def evaluate_latency_benchmarks(
        self, num_samples: int = 100
    ) -> Dict[str, float]:
        """Evaluate model latency across different scenarios."""
        latencies = {"simple_instruction": [], "tool_usage": [], "multi_step": []}

        # Sample different types of instructions
        simple_samples = [
            s for s in self.eval_dataset if s.get("complexity") == "simple"
        ][: num_samples // 3]
        tool_samples = [s for s in self.eval_dataset if s.get("tools_used")][
            : num_samples // 3
        ]
        complex_samples = [
            s for s in self.eval_dataset if s.get("complexity") == "complex"
        ][: num_samples // 3]

        # Measure latencies
        for sample in simple_samples:
            start_time = asyncio.get_event_loop().time()
            await self.model_handler.generate_response(sample["instruction"])
            latency = asyncio.get_event_loop().time() - start_time
            latencies["simple_instruction"].append(latency)

        for sample in tool_samples:
            start_time = asyncio.get_event_loop().time()
            await self.model_handler.generate_response(sample["instruction"])
            latency = asyncio.get_event_loop().time() - start_time
            latencies["tool_usage"].append(latency)

        for sample in complex_samples:
            start_time = asyncio.get_event_loop().time()
            await self.model_handler.generate_response(sample["instruction"])
            latency = asyncio.get_event_loop().time() - start_time
            latencies["multi_step"].append(latency)

        return {
            scenario: np.mean(times) if times else 0.0
            for scenario, times in latencies.items()
        }

    def generate_evaluation_report(self, results: EvaluationResult) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("Agent Model Evaluation Report")
        report.append("=" * 60)
        report.append("")

        report.append("OVERALL PERFORMANCE")
        report.append("-" * 30)
        report.append(f"Total Samples Evaluated: {results.total_samples}")
        report.append(f"Tool Selection Accuracy: {results.tool_selection_accuracy:.3f}")
        report.append(
            f"Parameter Extraction Accuracy: {results.parameter_extraction_accuracy:.3f}"
        )
        report.append(f"Task Completion Rate: {results.task_completion_rate:.3f}")
        report.append(f"Response Quality Score: {results.response_quality_score:.3f}")
        report.append("")

        report.append("ROUGE SCORES")
        report.append("-" * 30)
        for metric, score in results.rouge_scores.items():
            report.append(f"{metric.upper()}: {score:.3f}")
        report.append("")

        report.append("RELIABILITY METRICS")
        report.append("-" * 30)
        report.append(f"Hallucination Rate: {results.hallucination_rate:.3f}")
        report.append(f"Average Response Time: {results.average_response_time:.3f}s")
        report.append("")

        report.append("SUCCESS CRITERIA")
        report.append("-" * 30)
        criteria = [
            ("Tool Selection Accuracy > 85%", results.tool_selection_accuracy > 0.85),
            (
                "Parameter Extraction > 90%",
                results.parameter_extraction_accuracy > 0.90,
            ),
            ("Task Completion Rate > 80%", results.task_completion_rate > 0.80),
            ("Hallucination Rate < 10%", results.hallucination_rate < 0.10),
            ("Response Time < 5s", results.average_response_time < 5.0),
        ]

        for criterion, met in criteria:
            status = "✓ PASS" if met else "✗ FAIL"
            report.append(f"{criterion}: {status}")

        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)

        if results.tool_selection_accuracy < 0.85:
            report.append(
                "• Consider adding more diverse tool usage examples to training data"
            )

        if results.parameter_extraction_accuracy < 0.90:
            report.append("• Improve parameter validation in training examples")

        if results.hallucination_rate > 0.10:
            report.append("• Add negative examples to reduce hallucination")

        if results.average_response_time > 5.0:
            report.append("• Consider model optimization or hardware upgrade")

        return "\n".join(report)


class EvaluationDatasetGenerator:
    """Generate evaluation datasets for testing specific capabilities."""

    def __init__(self, output_dir: str = "./data/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_tool_accuracy_dataset(
        self, num_samples: int = 200
    ) -> List[Dict[str, Any]]:
        """Generate dataset specifically for testing tool selection accuracy."""
        samples = []

        # Single tool scenarios
        tool_scenarios = {
            "web_search": [
                "Find information about machine learning trends",
                "Search for the latest news about AI",
                "Look up restaurants in San Francisco",
            ],
            "calculator": [
                "Calculate 15% tip on $85.50",
                "What is 45 * 67 + 123?",
                "Find the area of a circle with radius 5",
            ],
            "weather": [
                "What's the weather like in Tokyo?",
                "Check the forecast for New York tomorrow",
                "Is it going to rain in Seattle this week?",
            ],
        }

        for tool, scenarios in tool_scenarios.items():
            for scenario in scenarios:
                for _ in range(num_samples // (len(tool_scenarios) * len(scenarios))):
                    samples.append(
                        {
                            "instruction": scenario,
                            "input": "",
                            "tools_used": [tool],
                            "complexity": "simple",
                            "expected_parameters": {
                                tool: self._generate_expected_params(tool, scenario)
                            },
                            "expected_keywords": self._generate_keywords(
                                tool, scenario
                            ),
                        }
                    )

        return samples

    def _generate_expected_params(self, tool: str, scenario: str) -> Dict[str, Any]:
        """Generate expected parameters for a tool based on scenario."""
        param_templates = {
            "web_search": {"query": "extracted_query", "max_results": 5},
            "calculator": {"expression": "extracted_expression"},
            "weather": {"location": "extracted_location"},
        }
        return param_templates.get(tool, {})

    def _generate_keywords(self, tool: str, scenario: str) -> List[str]:
        """Generate expected keywords in the response."""
        keyword_templates = {
            "web_search": ["search", "results", "found"],
            "calculator": ["calculate", "result", "equals"],
            "weather": ["weather", "temperature", "forecast"],
        }
        return keyword_templates.get(tool, [])
