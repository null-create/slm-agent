"""
Model handler for fine-tuned agent inference.
"""

import os
import json
import logging
import asyncio
from typing import Optional, Any
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from .mcp_client import MCPClient, MCPServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    AgentModelHandler is a class for managing the lifecycle and inference of a language model agent,
    optionally with LoRA adapters and tool integration via an MCP client.

    This handler is responsible for:
    - Loading a base language model and tokenizer from a specified path.
    - Optionally loading a LoRA adapter for fine-tuned weights.
    - Managing device placement, torch dtype, and trust settings.
    - Formatting prompts for chat and instruction-following tasks, including chat history.
    - Generating responses, with support for tool usage via an MCP client, including iterative tool calls.
    - Providing both synchronous and asynchronous batch generation interfaces.
    - Returning detailed model and environment information for diagnostics.

    Attributes:
        base_model_path (str): Path to the base model directory.
        adapter_path (str, optional): Path to the LoRA adapter directory.
        mcp_client (MCPClient): Client for tool integration and tool call execution.
        device (str): Device mapping for model placement (e.g., "cpu", "cuda", "auto").
        torch_dtype (str): Torch data type for model weights (e.g., "float16", "auto").
        trust_remote_code (bool): Whether to trust remote code execution for model/tokenizer.
        padding_side (str): Padding side for the tokenizer ("left" or "right").
        attn_implementation (str): Attention implementation for model loading.
        sys_prompt (str): System prompt loaded from file, used as context for all generations.
        tokenizer (AutoTokenizer): Tokenizer instance for the model.
        model (AutoModelForCausalLM or PeftModel): Loaded model instance.
        generation_config (GenerationConfig): Default generation configuration.
        pipeline (transformers.Pipeline): Text generation pipeline for easier inference.
        logger (logging.Logger): Logger for status and error reporting.

    Methods:
        __init__(...): Initialize the handler, load model/tokenizer, and set up logging.
        _load_model(): Internal method to load model, tokenizer, adapter, and system prompt.
        _initialized(): Check if model and tokenizer are loaded.
        format_prompt(instruction, input_text="", chat_history=None): Format a prompt for the model.
        async chat(): Interactive chat loop for user input and model response.
        async generate_response(...): Generate a response, optionally using tools via MCP client.
        async _generate_text(...): Generate text from the model given an instruction and input.
        _format_tool_results(tool_results): Format tool results for prompt inclusion.
        async batch_generate_async(...): Asynchronously generate responses for a batch of instructions.
        batch_generate(...): Synchronous wrapper for batch generation.
        get_model_info(): Return a dictionary with model and environment information.
        __repr__(): String representation of the handler instance.
    """

    def __init__(
        self,
        base_model_path: str,
        mcp_client: MCPClient,
        adapter_path: str = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
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
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.mcp_client = mcp_client
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.padding_side = padding_side
        self.attn_implementation = attn_implementation
        self.sys_prompt: str = None

        # Set random seed for reproducibility
        torch.random.manual_seed(random_seed)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.sys_prompt: str = None
        self.generation_config: GenerationConfig = None
        self.pipeline: pipeline = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self) -> None:
        """Load the fine-tuned model and tokenizer."""
        try:
            self.logger.info(f"Loading tokenizer from: {self.base_model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=self.trust_remote_code,
                padding_side=self.padding_side,
            )

            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.logger.info(f"Loading base model from: {self.base_model_path}")

            # Prepare model loading arguments
            model_args = {
                "dtype": self.torch_dtype,
                "device_map": self.device,
                "trust_remote_code": self.trust_remote_code,
                "attn_implementation": self.attn_implementation,
            }

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, **model_args
            )

            # Load LoRA adapter for fine-tune models if available, otherwise continue with base model
            if self.adapter_path:
                self.logger.info(f"Loading LoRA adapter from: {self.base_model_path}")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                self.logger.info("Using base model")
                self.model = base_model

            # Set the model to evaluation mode
            self.model.eval()

            # Set up generation configuration with better defaults
            self.generation_config = GenerationConfig(
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

            # Load the base system prompt
            prompt_file = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "sys_prompt.txt"
            )
            if not os.path.exists(prompt_file):
                raise FileNotFoundError(f"{prompt_file} not found!")

            with open(prompt_file, "r") as f:
                self.sys_prompt = f.read()

            # Initialize pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            self.logger.info("Model loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _initialized(self) -> bool:
        """Check if the model and tokenizer are properly initialized."""
        return self.model is not None and self.tokenizer is not None

    def format_prompt(
        self,
        instruction: str,
        input_text: str = "",
        chat_history: list[dict[str, str]] = None,
    ) -> str:
        """
        Format the prompt for the fine-tuned model.

        Args:
            instruction: The main instruction/query
            input_text: Optional additional input context

        Returns:
            Formatted prompt string
        """
        # Keep our chat history manageable so we don't quickly fill the
        # model's context window
        if chat_history and len(chat_history) > 50:
            chat_history = chat_history[-50:]

        if input_text:
            if chat_history:
                history_str = "\n".join(
                    [
                        f"User: {h['user']}\nAssistant: {h['assistant']}"
                        for h in chat_history
                    ]
                )
                full_prompt = (
                    f"{self.sys_prompt}\n\n### Conversation History:\n{history_str}\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    f"### Response:\n"
                )
            else:
                full_prompt = (
                    f"{self.sys_prompt}\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    f"### Response:\n"
                )
        elif chat_history:
            history_str = "\n".join(
                [
                    f"User: {h['user']}\nAssistant: {h['assistant']}"
                    for h in chat_history
                ]
            )
            full_prompt = (
                f"{self.sys_prompt}\n\n### Conversation History:\n{history_str}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )
        else:
            full_prompt = (
                f"{self.sys_prompt}\n\n### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

        return full_prompt

    async def chat(self) -> None:
        """
        Handle a chat message from the user.

        Args:
            user_message: The message sent by the user

        Returns:
            The assistant's response
        """
        while True:
            message = input("> ")
            if message.lower() in ["exit", "quit", "q"]:
                return

            response = await self.generate_response(
                instruction=message, generation_params=self.generation_config
            )
            print(f"🤖: {response['final_response']}\n")

    async def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        generation_params: Optional[GenerationParams] = None,
        max_tool_iterations: int = 3,
        use_pipeline: bool = True,
    ) -> dict[str, Any]:
        """
        Generate response with tool usage capability.

        Args:
            instruction: The main instruction/query
            input_text: Optional additional input context
            generation_params: Generation parameters
            max_tool_iterations: Maximum number of tool usage iterations
            use_pipeline: Whether to use the transformers pipeline for generation

        Returns:
            dictionary containing the response and metadata
        """
        if not self._initialized():
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
                    current_instruction, current_input, generation_params, use_pipeline
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

                    # Generate error response
                    error_response = await self._generate_text(
                        error_msg, current_input, generation_params, use_pipeline
                    )

                    return {
                        "final_response": error_response,
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
        self,
        instruction: str,
        input_text: str,
        generation_params: GenerationParams,
        use_pipeline: bool = False,
    ) -> str:
        """
        Generate text using the model.

        Args:
            instruction: The instruction/query
            input_text: Additional input context
            generation_params: Generation parameters
            use_pipeline: Whether to use the transformers pipeline for generation

        Returns:
            Generated response text
        """

        if not self._initialized():
            raise RuntimeError("Model and/or tokenizer not initialized")

        prompt = self.format_prompt(instruction, input_text)

        if use_pipeline:
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
                output = self.pipeline(prompt, **generation_args)
                response = output[0]["generated_text"].strip()
                return response

            except Exception as e:
                self.logger.error(f"Pipeline generation failed: {str(e)}")
                raise

        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=generation_params.max_new_tokens,
                    do_sample=generation_params.do_sample,
                    temperature=generation_params.temperature,
                    top_p=generation_params.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the response part
            if prompt in generated_text:
                return generated_text[len(prompt) :].strip()
            else:
                return generated_text.strip()

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
            "base_model_path": self.base_model_path,
            "adapter_path": self.adapter_path,
            "device": str(self.model.device) if self.model else None,
            "model_dtype": str(self.model.dtype) if self.model else None,
            "torch_dtype": str(self.torch_dtype),
            "vocab_size": len(self.tokenizer) if self.tokenizer else None,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
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
            f"base_model='{self.base_model_path}', "
            f"adapter='{self.adapter_path}', "
            f"device='{self.device}')"
        )


def setup_model_handler(
    model_info_file: str, base_model: Optional[str] = None
) -> AgentModelHandler:
    """
    Setup the model handler from model info file.

    Model file should contain an object with keys:
    {
        "base_model": "microsoft/Phi-3.5-mini-instruct",
        "adapter": null,
        "dtype": "bfloat16",
        "device": "auto",
        "tokenizer": "/path/to/tokenizer.json",
        "model-dir": "/path/to/model",
        "model-file": "/path/to/model/file"
    }

    Args:
        model_info_file: Path to the model info JSON file (model.json)
        base_model: Optional base model path override

    Returns:
        AgentModelHandler instance
    """
    with open(model_info_file, "r") as f:
        model_info: dict = json.load(f)

    logger.info(f"Load model info from {model_info_file}: {model_info}")
    logger.info("Initializing model...")
    try:
        mcp_server = MCPServer(
            name="slm-mcp-server",
            base_url=model_info.get("mcp-server", "http://localhost:9000/mcp"),
            description="MCP server for tool integration",
        )
        return AgentModelHandler(
            base_model_path=base_model or model_info.get("base_model"),
            mcp_client=MCPClient(servers=[mcp_server]),
            adapter_path=model_info.get("adapter"),
            device=model_info.get("device"),
            torch_dtype=model_info.get("dtype"),
        )
    except Exception as e:
        logger.error(f"Failed to setup model handler: {e}")
        raise


# TODO: add Flask app to serve model over specified endpoints so it can interact over the network

if __name__ == "__main__":
    from model_configs import ModelConfig

    model = setup_model_handler(ModelConfig.MODEL_PATH)

    asyncio.run(model.chat())
