"""
Model handler for fine-tuned agent inference.
"""

import logging
import asyncio
from typing import Optional, Any
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from .mcp_client import MCPClient


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
    """
    Handler for agent model inference with tool integration.

    This class provides an interface for loading fine-tuned models with LoRA adapters,
    formatting prompts, and generating responses with tool usage capability.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        mcp_client: MCPClient,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        random_seed: int = 42,
        padding_side: str = "left",
        attn_implementation: str = "eager",
    ) -> None:
        """
        Initialize the agent model handler.

        Args:
            base_model_path: Path to the base model
            adapter_path: Path to the LoRA adapter
            mcp_client: MCP client for tool integration
            device: Device mapping for model placement
            torch_dtype: Torch data type for model weights
            trust_remote_code: Whether to trust remote code execution
            random_seed: Random seed for reproducibility
            padding_side: Padding side for tokenizer
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.mcp_client = mcp_client
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.padding_side = padding_side
        self.attn_implementation = attn_implementation

        # Set random seed for reproducibility
        torch.random.manual_seed(random_seed)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.pipeline = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the fine-tuned model and tokenizer."""
        try:
            self.logger.info(f"Loading tokenizer from: {self.base_model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side=self.padding_side,
            )

            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info(f"Loading base model from: {self.base_model_name}")

            # Prepare model loading arguments
            model_args = {
                "dtype": self.torch_dtype,
                "device_map": self.device,
                "trust_remote_code": self.trust_remote_code,
                "attn_implementation": self.attn_implementation,
            }

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, **model_args
            )

            self.logger.info(f"Loading LoRA adapter from: {self.base_model_name}")

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()

            # Set up generation configuration with better defaults
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

            # Initialize pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            self.logger.info("Model loaded successfully!")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """
        Format the prompt for the fine-tuned model.

        Args:
            instruction: The main instruction/query
            input_text: Optional additional input context

        Returns:
            Formatted prompt string
        """
        try:
            tools_description = self.mcp_client.get_available_tools_description()
        except Exception as e:
            self.logger.warning(f"Failed to get tools description: {e}")
            tools_description = "No tools available."

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
        """
        Generate response with tool usage capability.

        Args:
            instruction: The main instruction/query
            input_text: Optional additional input context
            generation_params: Generation parameters
            max_tool_iterations: Maximum number of tool usage iterations

        Returns:
            dictionary containing the response and metadata
        """
        if not self.model or not self.tokenizer or not self.pipeline:
            raise RuntimeError("Model components not properly initialized")

        if generation_params is None:
            generation_params = GenerationParams()

        conversation_history = []
        current_instruction = instruction
        current_input = input_text

        for iteration in range(max_tool_iterations):
            self.logger.info(
                f"Generation iteration {iteration + 1}/{max_tool_iterations}"
            )

            try:
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
                    result
                    for result in tool_results
                    if not result.get("success", False)
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
                current_instruction = (
                    f"Based on the following tool results, provide a comprehensive response:\n\n"
                    f"{tool_results_text}\n\nOriginal request: {instruction}"
                )
                current_input = ""

            except Exception as e:
                self.logger.error(
                    f"Error in generation iteration {iteration + 1}: {str(e)}"
                )
                return {
                    "final_response": f"An error occurred during generation: {str(e)}",
                    "tool_calls_made": 0,
                    "iterations": iteration + 1,
                    "conversation_history": conversation_history,
                    "success": False,
                    "errors": [str(e)],
                }

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
        """
        Generate text using the pipeline.

        Args:
            instruction: The instruction/query
            input_text: Additional input context
            generation_params: Generation parameters

        Returns:
            Generated response text
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized")

        prompt = self.format_prompt(instruction, input_text)

        # Prepare generation arguments for pipeline
        generation_args = {
            "max_new_tokens": generation_params.max_new_tokens,
            "temperature": generation_params.temperature,
            "top_p": generation_params.top_p,
            "do_sample": generation_params.do_sample,
            "repetition_penalty": generation_params.repetition_penalty,
            "return_full_text": False,  # Only return generated text, not the prompt
            "pad_token_id": generation_params.pad_token_id
            or self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Add top_k if specified
        if generation_params.top_k is not None:
            generation_args["top_k"] = generation_params.top_k

        try:
            # Use pipeline for generation
            output = self.pipeline(prompt, **generation_args)
            response = output[0]["generated_text"].strip()
            return response

        except Exception as e:
            self.logger.error(f"Pipeline generation failed: {str(e)}")
            raise

    def _format_tool_results(self, tool_results: list[dict[str, Any]]) -> str:
        """
        Format tool results for inclusion in next iteration.

        Args:
            tool_results: list of tool execution results

        Returns:
            Formatted tool results string
        """
        formatted_results = []

        for result in tool_results:
            if result.get("success", False):
                tool_name = result.get("tool", "unknown")
                tool_result = result.get("result", {})

                formatted_results.append(f"Tool: {tool_name}")
                formatted_results.append(f"Result: {tool_result}")
                formatted_results.append("")

        return "\n".join(formatted_results)

    async def batch_generate_async(
        self,
        instructions: list[str],
        input_texts: Optional[list[str]] = None,
        generation_params: Optional[GenerationParams] = None,
        max_tool_iterations: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for multiple instructions asynchronously.

        Args:
            instructions: list of instructions/queries
            input_texts: Optional list of input texts
            generation_params: Generation parameters
            max_tool_iterations: Maximum tool iterations per request

        Returns:
            list of response dictionaries
        """
        if input_texts is None:
            input_texts = [""] * len(instructions)

        if len(instructions) != len(input_texts):
            raise ValueError("Instructions and input_texts must have the same length")

        # Process sequentially for now
        # In production, you might want to implement true parallel processing
        results = []
        for i, (instruction, input_text) in enumerate(zip(instructions, input_texts)):
            self.logger.info(f"Processing batch item {i + 1}/{len(instructions)}")
            try:
                result = await self.generate_response(
                    instruction, input_text, generation_params, max_tool_iterations
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch item {i + 1} failed: {str(e)}")
                results.append(
                    {
                        "final_response": f"Batch processing failed: {str(e)}",
                        "tool_calls_made": 0,
                        "iterations": 0,
                        "conversation_history": [],
                        "success": False,
                        "errors": [str(e)],
                    }
                )

        return results

    def batch_generate(
        self,
        instructions: list[str],
        input_texts: Optional[list[str]] = None,
        generation_params: Optional[GenerationParams] = None,
        max_tool_iterations: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Synchronous wrapper for batch generation.

        Args:
            instructions: list of instructions/queries
            input_texts: Optional list of input texts
            generation_params: Generation parameters
            max_tool_iterations: Maximum tool iterations per request

        Returns:
            list of response dictionaries
        """
        return asyncio.run(
            self.batch_generate_async(
                instructions, input_texts, generation_params, max_tool_iterations
            )
        )

    def get_model_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about the loaded model.

        Returns:
            dictionary containing model information
        """
        info = {
            "base_model_path": self.base_model_name,
            "adapter_path": self.adapter_path,
            "device": str(self.model.device) if self.model else None,
            "model_dtype": str(self.model.dtype) if self.model else None,
            "torch_dtype": str(self.torch_dtype),
            "vocab_size": len(self.tokenizer) if self.tokenizer else None,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "pipeline_loaded": self.pipeline is not None,
            "padding_side": self.padding_side,
        }

        # Add available tools if MCP client is available
        try:
            info["available_tools"] = list(self.mcp_client.available_tools.keys())
            info["tools_count"] = len(self.mcp_client.available_tools)
        except Exception as e:
            self.logger.warning(f"Could not get available tools: {e}")
            info["available_tools"] = []
            info["tools_count"] = 0

        # Add model config if available
        if self.model and hasattr(self.model, "config"):
            try:
                config = self.model.config
                info["model_config"] = {
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_attention_heads": getattr(config, "num_attention_heads", None),
                    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                    "max_position_embeddings": getattr(
                        config, "max_position_embeddings", None
                    ),
                    "model_type": getattr(config, "model_type", None),
                }
            except Exception as e:
                self.logger.warning(f"Could not get model config: {e}")

        return info

    def __repr__(self) -> str:
        return (
            f"AgentModelHandler("
            f"base_model='{self.base_model_name}', "
            f"adapter='{self.adapter_path}', "
            f"device='{self.device}')"
        )
