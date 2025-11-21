"""
HTTP server for serving the SLM agent model.
Provides REST API endpoints for chat interactions and model management.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid

from model_handler import AgentModelHandler, GenerationParams, setup_model_handler
from model_configs import ModelConfig

cfgs = ModelConfig()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PORT = cfgs.MODEL_PORT
HOST = cfgs.MODEL_HOST


class AgentServer:
    """
    Flask-based HTTP server for the SLM agent model.

    Provides endpoints for:
    - Chat message handling
    - Session management
    - Model information retrieval
    - Health checks
    """

    def __init__(
        self,
        model_handler: AgentModelHandler,
        host: str = HOST,
        port: int = PORT,
        debug: bool = False,
    ):
        """
        Initialize the agent server.

        Args:
            model_handler: Initialized AgentModelHandler instance
            host: Server host address
            port: Server port number
            debug: Enable debug mode
        """
        self.model_handler = model_handler
        self.host = host
        self.port = port
        self.debug = debug

        # Session storage (in production, use Redis or database)
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes

        # Register routes
        self._register_routes()

        logger.info(f"Agent server initialized on {host}:{port}")

    def _register_routes(self):
        """Register all API routes."""

        @self.app.route("/health", methods=["GET"])
        def health_check():
            """Health check endpoint."""
            return jsonify(
                {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_loaded": self.model_handler._initialized(),
                }
            )

        @self.app.route("/info", methods=["GET"])
        def model_info():
            """Get model information."""
            try:
                info = self.model_handler.get_model_info()
                return jsonify({"success": True, "data": info})
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/chat", methods=["POST"])
        def chat():
            """
            Handle chat message.

            Request body:
            {
                "message": "User message",
                "session_id": "optional-session-id",
                "input_text": "optional-additional-context",
                "generation_params": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    ...
                },
                "max_tool_iterations": 3,
                "use_pipeline": true
            }
            """
            try:
                data = request.get_json()

                if not data or "message" not in data:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Missing required field: message",
                            }
                        ),
                        400,
                    )

                message = data["message"]
                session_id = data.get("session_id")
                input_text = data.get("input_text", "")
                max_tool_iterations = data.get("max_tool_iterations", 3)
                use_pipeline = data.get("use_pipeline", True)

                # Parse generation parameters
                gen_params_dict = data.get("generation_params", {})
                gen_params = GenerationParams(
                    max_new_tokens=gen_params_dict.get("max_new_tokens", 512),
                    temperature=gen_params_dict.get("temperature", 0.7),
                    top_p=gen_params_dict.get("top_p", 0.9),
                    top_k=gen_params_dict.get("top_k", 50),
                    do_sample=gen_params_dict.get("do_sample", True),
                    repetition_penalty=gen_params_dict.get("repetition_penalty", 1.1),
                )

                # Create new session if needed
                if not session_id:
                    session_id = str(uuid.uuid4())
                    self.sessions[session_id] = {
                        "created_at": datetime.utcnow().isoformat(),
                        "message_count": 0,
                        "history": [],
                    }

                # Generate response
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(
                        self.model_handler.generate_response(
                            instruction=message,
                            input_text=input_text,
                            generation_params=gen_params,
                            max_tool_iterations=max_tool_iterations,
                            use_pipeline=use_pipeline,
                        )
                    )
                finally:
                    loop.close()

                # Update session
                if session_id in self.sessions:
                    self.sessions[session_id]["message_count"] += 1
                    self.sessions[session_id]["history"].append(
                        {
                            "user": message,
                            "assistant": response["final_response"],
                            "timestamp": datetime.utcnow().isoformat(),
                            "tool_calls_made": response.get("tool_calls_made", 0),
                            "iterations": response.get("iterations", 1),
                        }
                    )

                return jsonify(
                    {
                        "success": True,
                        "session_id": session_id,
                        "response": response["final_response"],
                        "metadata": {
                            "tool_calls_made": response.get("tool_calls_made", 0),
                            "iterations": response.get("iterations", 1),
                            "success": response.get("success", True),
                        },
                    }
                )

            except Exception as e:
                logger.error(f"Error in chat endpoint: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route("/chat/stream", methods=["POST"])
        def chat_stream():
            """
            Stream chat response (for future streaming support).
            Currently returns the full response.
            """
            # For now, redirect to regular chat
            # In future, implement true streaming with Server-Sent Events
            return chat()

        @self.app.route("/session/<session_id>", methods=["GET"])
        def get_session(session_id):
            """Get session information."""
            if session_id not in self.sessions:
                return jsonify({"success": False, "error": "Session not found"}), 404

            return jsonify({"success": True, "session": self.sessions[session_id]})

        @self.app.route("/session/<session_id>", methods=["DELETE"])
        def delete_session(session_id):
            """Delete a session."""
            if session_id in self.sessions:
                del self.sessions[session_id]
                return jsonify({"success": True, "message": "Session deleted"})

            return jsonify({"success": False, "error": "Session not found"}), 404

        @self.app.route("/sessions", methods=["GET"])
        def list_sessions():
            """List all active sessions."""
            return jsonify(
                {
                    "success": True,
                    "sessions": {
                        sid: {
                            "created_at": data["created_at"],
                            "message_count": data["message_count"],
                        }
                        for sid, data in self.sessions.items()
                    },
                }
            )

        @self.app.route("/batch", methods=["POST"])
        def batch_generate():
            """
            Handle batch generation.

            Request body:
            {
                "instructions": ["message1", "message2", ...],
                "input_texts": ["context1", "context2", ...],  // optional
                "generation_params": {...},  // optional
                "max_tool_iterations": 3  // optional
            }
            """
            try:
                data = request.get_json()

                if not data or "instructions" not in data:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Missing required field: instructions",
                            }
                        ),
                        400,
                    )

                instructions = data["instructions"]
                input_texts = data.get("input_texts")
                max_tool_iterations = data.get("max_tool_iterations", 3)

                # Parse generation parameters
                gen_params_dict = data.get("generation_params", {})
                gen_params = GenerationParams(
                    max_new_tokens=gen_params_dict.get("max_new_tokens", 512),
                    temperature=gen_params_dict.get("temperature", 0.7),
                    top_p=gen_params_dict.get("top_p", 0.9),
                    top_k=gen_params_dict.get("top_k", 50),
                    do_sample=gen_params_dict.get("do_sample", True),
                    repetition_penalty=gen_params_dict.get("repetition_penalty", 1.1),
                )

                # Generate responses
                results = self.model_handler.batch_generate(
                    instructions=instructions,
                    input_texts=input_texts,
                    generation_params=gen_params,
                    max_tool_iterations=max_tool_iterations,
                )

                return jsonify(
                    {"success": True, "results": results, "count": len(results)}
                )

            except Exception as e:
                logger.error(f"Error in batch endpoint: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), 500

    def run(self):
        """Start the Flask server."""
        logger.info(f"Starting agent server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug, threaded=True)


def create_server(
    model_info_file: str,
    base_model: Optional[str] = None,
    host: str = HOST,
    port: int = PORT,
    debug: bool = False,
) -> AgentServer:
    """
    Create and configure the agent server.

    Args:
        model_info_file: Path to model info JSON file
        base_model: Optional base model path override
        host: Server host address
        port: Server port number
        debug: Enable debug mode

    Returns:
        Configured AgentServer instance
    """
    logger.info("Setting up model handler...")
    model_handler = setup_model_handler(model_info_file, base_model)

    logger.info("Creating server...")
    server = AgentServer(model_handler=model_handler, host=host, port=port, debug=debug)

    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLM Agent HTTP Server")
    parser.add_argument(
        "--model-info",
        type=str,
        required=True,
        help="Path to model info JSON file",
        default=cfgs.MODEL_META_DATA,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Override base model path",
        default=cfgs.MODEL_PATH,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
        help=f"Server host address (default: '{HOST}')",
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Server port number (default: {PORT}"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode", default=True
    )

    args = parser.parse_args()

    server = create_server(
        model_info_file=args.model_info,
        base_model=args.base_model,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    server.run()
