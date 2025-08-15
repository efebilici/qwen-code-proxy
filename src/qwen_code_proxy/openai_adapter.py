"""
OpenAI adapter module

Handles format conversion and compatibility
"""

import time
import uuid
import logging
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage
)
from .qwen_client import QwenClient

logger = logging.getLogger(__name__)

# Initialize QwenClient
qwen_client = QwenClient()


class OpenAIAdapter:
    """OpenAI format adapter"""

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Handle chat completion request (non-streaming)
        
        Args:
            request: OpenAI format chat completion request
            
        Returns:
            OpenAI format chat completion response
        """
        logger.info(
            "Processing chat completion request, model: %s, messages: %d",
            request.model,
            len(request.messages),
        )

        try:
            # Prepare request data for Qwen API
            request_data = {
                "messages": [msg.dict() for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            }

            # Call QwenClient to make the request
            response_data = await qwen_client.make_request(request_data)

            # Extract the response text
            response_text = response_data["choices"][0]["message"]["content"]

            # Build OpenAI format response
            response = ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
            )

            logger.info(
                "Chat completion request processed successfully, response length: %d",
                len(response_text),
            )
            return response

        except Exception as e:
            logger.error(f"Error processing chat completion request: {e}")
            raise

    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> StreamingResponse:
        """
        Handle streaming chat completion request
        
        Args:
            request: OpenAI format chat completion request
            
        Returns:
            Streaming response
        """
        logger.info(
            "Processing streaming chat completion request, model: %s, messages: %d",
            request.model,
            len(request.messages),
        )

        async def generate_stream() -> AsyncGenerator[str, None]:
            """Generate streaming response data"""
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_time = int(time.time())

            try:
                # Prepare request data for Qwen API
                request_data = {
                    "messages": [msg.dict() for msg in request.messages],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": True,
                }

                # Get streaming data generator
                stream_generator = await qwen_client.make_request(request_data)

                # Send data chunks one by one
                async for chunk in stream_generator:
                    stream_response = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"content": chunk},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {stream_response.model_dump_json()}\n\n"

                # Send end marker
                final_response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0, delta={}, finish_reason="stop"
                        )
                    ],
                )
                yield f"data: {final_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

                logger.info("Streaming chat completion request processed successfully")

            except Exception as e:
                logger.error(f"Error processing streaming chat completion request: {e}")
                # Send error information
                error_response = {
                    "error": {"message": str(e), "type": "internal_error"}
                }
                yield f"data: {error_response}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )


# Global adapter instance
openai_adapter = OpenAIAdapter() 