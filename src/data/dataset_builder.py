"""
Dataset builder for PHI-3.5 agentic fine-tuning.
Creates datasets with tool usage examples and MCP server interactions.
"""

import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AgenticExample:
    instruction: str
    input: str
    output: str
    tools_used: List[str]
    complexity: str  # "simple", "medium", "complex"


class AgenticDatasetBuilder:
    def __init__(self, output_dir: str = "./data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tool definitions for the agent
        self.available_tools = {
            "web_search": {
                "description": "Search the web for information",
                "parameters": ["query", "max_results"],
            },
            "file_reader": {
                "description": "Read and analyze files",
                "parameters": ["file_path", "operation"],
            },
        }

    def create_tool_usage_example(
        self, tool_name: str, scenario: str
    ) -> AgenticExample:
        """Create a single tool usage example."""
        tool_info = self.available_tools[tool_name]

        # Generate instruction based on tool and scenario
        instructions = {
            "web_search": f"Search for information about {scenario}",
            "file_reader": f"Read and analyze the file containing {scenario}",
        }

        instruction = instructions.get(
            tool_name, f"Use {tool_name} to help with {scenario}"
        )

        # Generate expected output with proper tool usage format
        output = self._generate_tool_response(tool_name, scenario, tool_info)

        return AgenticExample(
            instruction=instruction,
            input="",
            output=output,
            tools_used=[tool_name],
            complexity="simple",
        )

    def create_multi_step_example(
        self, scenario: str, tools: List[str]
    ) -> AgenticExample:
        """Create a multi-step agentic example using multiple tools."""
        instruction = f"Help me with this multi-step task: {scenario}"

        output_parts = [
            "I'll help you with this multi-step task. Let me break it down:\n"
        ]

        for i, tool in enumerate(tools, 1):
            tool_info = self.available_tools[tool]
            output_parts.append(
                f"Step {i}: I'll use {tool} to {tool_info['description'].lower()}"
            )
            output_parts.append(self._generate_tool_call(tool, scenario, tool_info))
            output_parts.append(
                f"Based on the {tool} results, I can now proceed to the next step.\n"
            )

        output_parts.append(
            "I've completed all the steps successfully. The task is now complete."
        )

        return AgenticExample(
            instruction=instruction,
            input="",
            output="\n".join(output_parts),
            tools_used=tools,
            complexity="complex",
        )

    def _generate_tool_call(
        self, tool_name: str, scenario: str, tool_info: Dict
    ) -> str:
        """Generate a properly formatted tool call."""
        # Sample parameters based on tool type
        params = self._generate_sample_parameters(tool_name, scenario)

        return f"""
<tool_use>
<tool_name>{tool_name}</tool_name>
<parameters>
{json.dumps(params, indent=2)}
</parameters>
</tool_use>
"""

    def _generate_tool_response(
        self, tool_name: str, scenario: str, tool_info: Dict
    ) -> str:
        """Generate a complete response with tool usage."""
        response_start = f"I'll use the {tool_name} tool to help with {scenario}.\n"
        tool_call = self._generate_tool_call(tool_name, scenario, tool_info)
        response_end = f"\nBased on the {tool_name} results, I've successfully completed your request."

        return response_start + tool_call + response_end

    def _generate_sample_parameters(
        self, tool_name: str, scenario: str
    ) -> Dict[str, Any]:
        """Generate realistic parameters for each tool type."""
        param_generators = {
            "web_search": lambda s: {"query": s, "max_results": 5},
            "file_reader": lambda s: {
                "file_path": f"/data/{s}.txt",
                "operation": "read",
            },
        }

        return param_generators.get(tool_name, lambda s: {})(scenario)

    def generate_dataset(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate a complete dataset for training."""
        examples = []

        # Single tool examples (60% of dataset)
        single_tool_count = int(num_examples * 0.6)
        for _ in range(single_tool_count):
            tool = random.choice(list(self.available_tools.keys()))
            scenario = self._generate_random_scenario(tool)
            example = self.create_tool_usage_example(tool, scenario)
            examples.append(self._to_dict(example))

        # Multi-step examples (40% of dataset)
        multi_step_count = num_examples - single_tool_count
        for _ in range(multi_step_count):
            num_tools = random.randint(2, 3)
            tools = random.sample(list(self.available_tools.keys()), num_tools)
            scenario = self._generate_multi_step_scenario()
            example = self.create_multi_step_example(scenario, tools)
            examples.append(self._to_dict(example))

        return examples

    def _generate_random_scenario(self, tool_name: str) -> str:
        """Generate random scenarios for different tools."""
        scenarios = {
            "web_search": [
                "latest AI research",
                "best restaurants in Seattle",
                "climate change effects",
            ],
            "file_reader": ["sales data", "log files", "configuration settings"],
        }

        return random.choice(scenarios.get(tool_name, ["general information"]))

    def _generate_multi_step_scenario(self) -> str:
        """Generate scenarios that require multiple tools."""
        scenarios = [
            "research and calculate the cost of living comparison between two cities",
            "read configuration files and calculate system performance metrics",
        ]
        return random.choice(scenarios)

    def _to_dict(self, example: AgenticExample) -> Dict[str, Any]:
        """Convert AgenticExample to dictionary format for training."""
        return {
            "instruction": example.instruction,
            "input": example.input,
            "output": example.output,
            "tools_used": example.tools_used,
            "complexity": example.complexity,
        }

    def save_dataset(
        self, examples: List[Dict[str, Any]], filename: str = "training_dataset.json"
    ):
        """Save dataset to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        print(f"Dataset saved to {filepath}")
        print(f"Total examples: {len(examples)}")

        # Print statistics
        complexities = [ex["complexity"] for ex in examples]
        tools_used = [len(ex["tools_used"]) for ex in examples]

        print(
            f"Complexity distribution: {dict(zip(*zip(*[(c, complexities.count(c)) for c in set(complexities)])))}"
        )
        print(f"Average tools per example: {sum(tools_used) / len(tools_used):.2f}")


if __name__ == "__main__":
    builder = AgenticDatasetBuilder()
    dataset = builder.generate_dataset(num_examples=5000)
    builder.save_dataset(dataset, "agentic_training_dataset.json")
