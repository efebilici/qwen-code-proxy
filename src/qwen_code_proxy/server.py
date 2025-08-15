"""
FastAPI server module

Implements HTTP service and API endpoints
"""

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from . import __version__
from .config import config
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    ModelsResponse,
    ModelInfo
)
from .qwen_client import QwenClient

# Configure logging (will be updated when config is set)
logging.basicConfig(
    level=logging.INFO,  # Default level, will be updated in CLI
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('qwen_code_proxy')

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Ensure logging level is applied after uvicorn starts
    import logging
    logging.getLogger('qwen_code_proxy').setLevel(getattr(logging, config.log_level.upper()))
    
    logger.info(f"Starting Qwen Code Proxy v{__version__}")
    logger.info(f"Configuration: port={config.port}, rate_limit={config.rate_limit}/min, concurrency={config.max_concurrency}")
    logger.debug(f"Debug logging is enabled (log_level={config.log_level})")
    yield
    logger.info("Shutting down Qwen Code Proxy")


# Create FastAPI application
app = FastAPI(
    title="Qwen Code Proxy",
    description="OpenAI-compatible API wrapper for Qwen",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception occurred while processing request: {exc}")
    logger.error(f"Exception details: {traceback.format_exc()}")
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            message="Internal server error",
            type="internal_error",
            code="500"
        )
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(version=__version__)


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    models = [
        ModelInfo(id=model_id) for model_id in config.supported_models
    ]
    
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@limiter.limit(f"{config.rate_limit}/minute")
async def chat_completions(
    chat_request: ChatCompletionRequest,
    request: Request
):
    """
    Chat completion endpoint
    
    Implements OpenAI-compatible chat completion API
    """
    logger.info(f"Received chat completion request: model={chat_request.model}, stream={chat_request.stream}")
    
    try:
        # Validate model
        if chat_request.model not in config.supported_models:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=f"Unsupported model: {chat_request.model}. Supported models: {', '.join(config.supported_models)}",
                        type="invalid_request_error",
                        param="model"
                    )
                ).model_dump()
            )
        
        qwen_client = QwenClient(model=chat_request.model)
        
        # Handle non-streaming request
        response_data = qwen_client.make_request(chat_request.model_dump(exclude={'model'}))
        
        # This is a simplified adapter logic. A more robust solution would
        # map the full Qwen response to the OpenAI format.
        return ChatCompletionResponse.model_validate(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat completion request: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message="Internal server error",
                    type="internal_error",
                    code="500"
                )
            ).model_dump()
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port) 