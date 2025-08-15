"""
CLI entry module

Uses Click to handle command line arguments and application startup
"""

import click
import uvicorn


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Server host address"
)
@click.option(
    "--port",
    default=8765,
    type=int,
    help="Server port"
)
@click.option(
    "--rate-limit",
    default=60,
    type=int,
    help="Maximum requests per minute"
)
@click.option(
    "--max-concurrency",
    default=4,
    type=int,
    help="Maximum concurrent subprocesses"
)
@click.option(
    "--timeout",
    default=30.0,
    type=float,
    help="Qwen CLI command timeout in seconds"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
def main(
    host: str,
    port: int,
    rate_limit: int,
    max_concurrency: int,
    timeout: float,
    debug: bool
):
    """Start Qwen Code Proxy server"""
    
    # Set configuration
    import os
    from .config import config
    
    # Set environment variable for reload mode
    os.environ['QWEN_CODE_PROXY_DEBUG'] = str(debug)
    
    config.host = host
    config.port = port
    config.log_level = "debug" if debug else "info"
    config.rate_limit = rate_limit
    config.max_concurrency = max_concurrency
    config.timeout = timeout
    config.debug = debug
    
    # Update logging level based on configuration
    import logging
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
    # Also set level for all qwen_code_proxy loggers
    logging.getLogger('qwen_code_proxy').setLevel(getattr(logging, config.log_level.upper()))
    
    # Start server
    uvicorn.run(
        "qwen_code_proxy.server:app",
        host=host,
        port=port,
        log_level=config.log_level,
        reload=debug
    )


if __name__ == "__main__":
    main() 