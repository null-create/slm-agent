"""
Model handler for fine-tuned agent inference.
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
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


class ModelHandler:
    """
    A handler class for the Microsoft Phi-3 Mini-4K-Instruct model.

    This class provides an easy-to-use interface for loading the model,
    formatting chat conversations, and generating responses.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device_map: str = "cuda",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        attn_implementation: Optional[str] = None,
        random_seed: int = 0,
    ):
        """
        Initialize model handler.

        Args:
            model_name: The model name/path to load
            device_map: Device mapping for model placement
            torch_dtype: Torch data type for model weights
            trust_remote_code: Whether to trust remote code execution
            attn_implementation: Attention implementation ("flash_attention_2" for flash attention)
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation

        # Set random seed for reproducibility
        torch.random.manual_seed(random_seed)

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the model, tokenizer, and initialize the pipeline."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")

            # Prepare model loading arguments
            model_args = {
                "device_map": self.device_map,
                "torch_dtype": self.torch_dtype,
                "trust_remote_code": self.trust_remote_code,
            }

            # Add flash attention if specified
            if self.attn_implementation:
                model_args["attn_implementation"] = self.attn_implementation

            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_args
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Initialize pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    @staticmethod
    def format_chat_prompt(
        system_message: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Format a chat prompt using Phi-3's chat format.

        Args:
            system_message: The system prompt
            user_message: The user's message
            conversation_history: Optional conversation history as list of dicts
                                with 'role' and 'content' keys

        Returns:
            Formatted chat prompt string
        """
        prompt = f"<|system|>\n{system_message}<|end|>\n"

        # Add conversation history if provided
        if conversation_history:
            for message in conversation_history:
                role = message.get("role", "")
                content = message.get("content", "")

                if role == "user":
                    prompt += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}<|end|>\n"

        # Add current user message
        prompt += f"<|user|>\n{user_message}<|end|>\n<|assistant|>\n"

        return prompt

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        do_sample: bool = False,
        return_full_text: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate a response using the chat format.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            return_full_text: Whether to return the full text including input
            **kwargs: Additional generation arguments

        Returns:
            Generated response text
        """
        if not self.pipeline:
            raise RuntimeError("Model pipeline not initialized")

        # Prepare generation arguments
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": return_full_text,
            "temperature": temperature,
            "do_sample": do_sample,
            **kwargs,
        }

        try:
            # Generate response
            output = self.pipeline(messages, **generation_args)
            return output[0]["generated_text"]

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise

    def chat(
        self,
        user_message: str,
        system_message: str = "You are a helpful AI assistant.",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **generation_kwargs,
    ) -> str:
        """
        Simple chat interface using string inputs.

        Args:
            user_message: The user's message
            system_message: The system prompt
            conversation_history: Optional conversation history
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated response text
        """
        # Build messages list
        messages = [{"role": "system", "content": system_message}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        return self.generate_response(messages, **generation_kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        info = {
            "model_name": self.model_name,
            "vocab_size": getattr(self.tokenizer, "vocab_size", None),
            "device_map": self.device_map,
            "torch_dtype": str(self.torch_dtype),
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "pipeline_loaded": self.pipeline is not None,
        }

        if hasattr(self.model, "config"):
            info["model_config"] = {
                "hidden_size": getattr(self.model.config, "hidden_size", None),
                "num_attention_heads": getattr(
                    self.model.config, "num_attention_heads", None
                ),
                "num_hidden_layers": getattr(
                    self.model.config, "num_hidden_layers", None
                ),
                "max_position_embeddings": getattr(
                    self.model.config, "max_position_embeddings", None
                ),
            }

        return info

    def __repr__(self) -> str:
        return f"ModelHandler(model_name='{self.model_name}', device_map='{self.device_map}')"
