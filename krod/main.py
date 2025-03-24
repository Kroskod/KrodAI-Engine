"""
KROD Main Entry Point
--------------------
Main entry point for the KROD AI research assistant.
Handles initialization, CLI bootstrapping, and configuration loading.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any
from dotenv import load_dotenv
from krod.interfaces.cli import KrodCLI
from krod.core.config import load_config, DEFAULT_CONFIG
from krod.core.engine import KrodEngine

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_level = config.get("log_level", "INFO").upper()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Set up logging to file if specified
    log_file = config.get("log_file")
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="KROD - Knowledge-Reinforced Operational Developer"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file",
        default=os.getenv("KROD_CONFIG")
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--query",
        help="Process a single query and exit"
    )
    
    parser.add_argument(
        "--session",
        help="Use specific session ID"
    )
    
    return parser.parse_args()

def initialize_krod(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Initialize KROD with configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.debug:
        config["debug"] = True
        config["log_level"] = "DEBUG"
    
    # Set up logging
    setup_logging(config)
    
    # Log initialization
    logging.info("Initializing KROD...")
    logging.debug("Configuration loaded: %s", config)
    
    return config

def display_welcome_message() -> None:
    """Display welcome message and version information."""
    welcome = """
    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║   Krod - Knowledge-Reinforced Operational Developer     ║
    ║                                                         ║
    ║   Version: 0.1.0                                        ║
    ║   Mode: CLI                                             ║
    ║                                                         ║
    ║   Type 'help' for a list of commands                    ║
    ║   Type 'quit' to exit                                   ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝
    """
    print(welcome)

def check_requirements() -> bool:
    """
    Check if all requirements are met.
    
    Returns:
        Boolean indicating if requirements are met
    """
    required_env_vars = ["ANTHROPIC_API_KEY"]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    return True

def main() -> None:
    """Main entry point for KROD."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Check requirements
        if not check_requirements():
            sys.exit(1)
        
        # Initialize KROD
        config = initialize_krod(args)
        
        # Display welcome message
        display_welcome_message()
        
        if args.query:
            # Process single query mode
            engine = KrodEngine(config)
            result = engine.process(args.query, args.session)
            print("\nResponse:")
            print("---------")
            print(result["response"])
            print("\nToken usage:", result.get("token_usage", 0))
        else:
            # Start interactive CLI mode
            cli = KrodCLI(config)
            
            # Set session if specified
            if args.session:
                session = cli.engine.research_context.get(args.session)
                if session:
                    cli.current_session_id = args.session
                    print(f"Using session: {args.session}")
                else:
                    print(f"Warning: Session not found: {args.session}")
            
            # Start CLI loop
            try:
                cli.cmdloop()
            except KeyboardInterrupt:
                print("\nExiting KROD. Goodbye!")
            finally:
                cli._save_history()
    
    except Exception as e:
        logging.error("Fatal error: %s", str(e), exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()