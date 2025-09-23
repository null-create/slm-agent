"""
Model handler for fine-tuned agent inference.
"""

import logging
import asyncio
from typing import Any, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from .mcp_client import MCPClient, ToolCall


@dataclass
class GenerationParams:
    """Parameters for text generation."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None


class AgentModelHandler:
    """Handler for
    model inference with tool integration."""

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        mcp_client: MCPClient,
        device: str = "auto",
    ) -> None:
        """Initialize the model handler."""
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.mcp_client = mcp_client
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.tokenizer = None
        self.model = None
        self.generation_config = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the fine-tuned model and tokenizer."""
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            # trust_remote_code=True,
            # padding_side="left",  # For batch generation
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            # torch_dtype=torch.bfloat16,
            # device_map=self.device,
            # trust_remote_code=True,
        )

        self.logger.info("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()

        # Set up generation configuration
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

        self.logger.info("Model loaded successfully!")

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format the prompt for the fine-tuned model."""
        tools_description = self.mcp_client.get_available_tools_description()

        # TODO: update the system_prompt and possibly store an external template
        # somewhere that's read in upon initialization

        system_prompt = f"""You are a helpful AI assistant that can use tools to complete tasks. {tools_description}

When you need to use a tool, format your response with the tool usage blocks as shown above. Always explain what you're doing and provide helpful responses based on the tool results."""

        if input_text:
            full_prompt = f"{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            full_prompt = (
                f"{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            )

        return full_prompt

    async def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        generation_params: Optional[GenerationParams] = None,
        max_tool_iterations: int = 3,
    ) -> dict[str, Any]:
        """Generate response with tool usage capability."""

        if generation_params is None:
            generation_params = GenerationParams()

        conversation_history = []
        current_instruction = instruction
        current_input = input_text

        for iteration in range(max_tool_iterations):
            self.logger.info(f"Generation iteration {iteration + 1}")

            # Generate text response
            response = await self._generate_text(
                current_instruction, current_input, generation_params
            )

            conversation_history.append(
                {
                    "iteration": iteration + 1,
                    "instruction": current_instruction,
                    "input": current_input,
                    "response": response,
                }
            )

            # Parse tool calls from response
            tool_calls = self.mcp_client.parse_tool_calls(response)

            if not tool_calls:
                # No tool calls found, return final response
                return {
                    "final_response": response,
                    "tool_calls_made": sum(
                        len(h.get("tool_results", [])) for h in conversation_history
                    ),
                    "iterations": iteration + 1,
                    "conversation_history": conversation_history,
                    "success": True,
                }

            # Execute tool calls
            self.logger.info(f"Executing {len(tool_calls)} tool calls")
            tool_results = await self.mcp_client.execute_tool_calls(tool_calls)

            conversation_history[-1]["tool_calls"] = [
                {"name": call.name, "parameters": call.parameters}
                for call in tool_calls
            ]
            conversation_history[-1]["tool_results"] = tool_results

            # Check if any tool calls failed
            failed_calls = [
                result for result in tool_results if not result.get("success", False)
            ]
            if failed_calls:
                error_msg = (
                    f"Tool execution failed: {[f['error'] for f in failed_calls]}"
                )
                self.logger.error(error_msg)

                return {
                    "final_response": f"I encountered errors while using tools: {error_msg}",
                    "tool_calls_made": len(tool_calls),
                    "iterations": iteration + 1,
                    "conversation_history": conversation_history,
                    "success": False,
                    "errors": failed_calls,
                }

            # Prepare next iteration with tool results
            tool_results_text = self._format_tool_results(tool_results)
            current_instruction = f"Based on the following tool results, provide a comprehensive response:\n\n{tool_results_text}\n\nOriginal request: {instruction}"
            current_input = ""

        # Max iterations reached
        final_response = (
            conversation_history[-1]["response"]
            if conversation_history
            else "Maximum iterations reached without completion."
        )

        return {
            "final_response": final_response,
            "tool_calls_made": sum(
                len(h.get("tool_results", [])) for h in conversation_history
            ),
            "iterations": max_tool_iterations,
            "conversation_history": conversation_history,
            "success": False,
            "message": "Maximum tool iterations reached",
        }

    async def _generate_text(
        self, instruction: str, input_text: str, generation_params: GenerationParams
    ) -> str:
        """Generate text using the model."""
        prompt = self.format_prompt(instruction, input_text)

        # Tokenize input
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_params.max_new_tokens,
                temperature=generation_params.temperature,
                top_p=generation_params.top_p,
                top_k=generation_params.top_k,
                do_sample=generation_params.do_sample,
                repetition_penalty=generation_params.repetition_penalty,
                pad_token_id=generation_params.pad_token_id
                or self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        return response

    def _format_tool_results(self, tool_results: list[dict[str, Any]]) -> str:
        """Format tool results for inclusion in next iteration."""
        formatted_results = []

        for result in tool_results:
            if result.get("success", False):
                tool_name = result.get("tool", "unknown")
                tool_result = result.get("result", {})

                formatted_results.append(f"Tool: {tool_name}")
                formatted_results.append(f"Result: {tool_result}")
                formatted_results.append("")

        return "\n".join(formatted_results)

    async def _batch_generate(
        self,
        instructions: list[str],
        input_texts: list[str] = None,
        generation_params: Optional[GenerationParams] = None,
    ) -> list[dict[str, Any]]:
        """Generate responses for multiple instructions."""

        if input_texts is None:
            input_texts = [""] * len(instructions)

        if len(instructions) != len(input_texts):
            raise ValueError("Instructions and input_texts must have the same length")

        # For simplicity, process sequentially
        # In production, you might want to implement true batching
        results = []
        for instruction, input_text in zip(instructions, input_texts):
            result = await self.generate_response(
                instruction, input_text, generation_params
            )
            results.append(result)

        return results

    # Synchronous wrapper for batch inference generation.
    # May be quite slow.
    def batch_generate(
        self,
        instructions: list[str],
        input_texts: list[str] = None,
        generation_params: Optional[GenerationParams] = None,
    ) -> list[dict[str, Any]]:
        return asyncio.run(
            self._batch_generate(instructions, input_texts, generation_params)
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "base_model": self.base_model_path,
            "adapter_path": self.adapter_path,
            "device": str(self.model.device) if self.model else None,
            "model_dtype": str(self.model.dtype) if self.model else None,
            "vocab_size": len(self.tokenizer) if self.tokenizer else None,
            "available_tools": list(self.mcp_client.available_tools.keys()),
        }
