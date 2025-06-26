import sys
import asyncio
import logging

from typing import Dict, Any
from rich import print
from rich.console import Console
from rich.panel import Panel
import uuid
from krod.core.engine import KrodEngine
from krod.interfaces.commands.vector_store import vector_store as vector_store_commands

class KrodCLI:
    """
    Command Line Interface for KROD.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the CLI."""
        self.config = config
        self.console = Console()
        self.conversation_history = []  # Store conversation history
        self.running = True
        self.user_id = "cli_user"  # Default user ID for CLI
        self.session_id = str(uuid.uuid4())  # Generate a new session ID
        
        try:
            self.engine = KrodEngine(self.config)
            self.logger = logging.getLogger("krod.cli")
        except Exception as e:
            print(f"Error initializing KROD engine: {str(e)}")
            sys.exit(1)

    def display_welcome(self):
        """Display welcome message."""
        welcome = """
    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║   Krod AI - Research Amplification Partner              ║
    ║                                                         ║
    ║   Version: 0.2.0                                        ║
    ║   Type 'help' for a list of commands                    ║
    ║   Type 'quit' to exit                                   ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝
        """
        print(welcome)

    # Process a command or query
    async def process_command(self, command: str) -> bool:
        """Process a command or query asynchronously."""
        command = command.strip()
        
        if not command:
            return True
            
        # Handle built-in commands
        if command.lower() in ('quit', 'exit'):
            self.running = False
            # Save history before exiting
            self._save_history()
            print("\nThank you for using KROD. Goodbye!")
            return False
            
        if command.lower() == 'help':
            self.show_help()
            return True

        if command.startswith('vector_store'):
            try:
                import shlex
                import click
                args = shlex.split(command)[1:]
                return vector_store_commands.main(args, standalone_mode=False) == 0
            except click.ClickException as e:
                e.show()
                return True
            except Exception as e:
                print(f"Error executing vector store command: {str(e)}", file=sys.stderr)
                return True
        
        # Process as a regular query with conversation memory
        try:
            response = await self.engine.process(
                query=command,
                context_id=self.session_id,
                conversation_history=self.conversation_history,
                user_id=self.user_id,
                session_id=self.session_id
            )
            
            # Update conversation history from response
            self.conversation_history = response.get("conversation_history", [])
            
            # Save history after processing command
            self._save_history()
            
            # Format and display the response
            self._display_response(response)
            
        except Exception as e:
            print(f"Error processing query: {str(e)}", file=sys.stderr)
            self.logger.error(f"Error in process_command: {str(e)}", exc_info=True)
            
        return True

    def _display_response(self, response: Dict[str, Any]) -> None:
        """Format and display the response to the user."""
        print("\n" + "=" * 80)
        
        # Display the main response
        if "response" in response:
            print(response["response"])
        
        # Display any metadata or sources if available
        if "sources" in response and response["sources"]:
            print("\nSources:")
            for i, source in enumerate(response["sources"], 1):
                print(f"  {i}. {source.get('title', 'No title')}")
                if 'url' in source:
                    print(f"     {source['url']}")
        
        print("=" * 80 + "\n")


    def show_help(self):
        """Show help information."""
        help_text = """
        Available Commands:
        ------------------
        help         : Show this help message
        quit/exit   : Exit KROD
        vector_store: Manage vector store operations
        
        You can type your queries directly at the prompt.
        
        Examples:
        ---------
        > analyze this code: def hello(): print("Hello")
        > solve this equation: 2x + 5 = 13
        > what is the complexity of quicksort?
        """
        print(help_text)

    def _save_history(self):
        """Save command history to a file."""
        import json
        import os
        from pathlib import Path
        from datetime import datetime, timezone
        
        try:
            # Create directory if it doesn't exist
            history_dir = Path(self.config.get("storage_path", "./data/history"))
            history_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with user_id and session_id
            filename = f"history_{self.user_id}_{self.session_id}.json"
            filepath = history_dir / filename
            
            # Prepare data to save
            history_data = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "conversation_history": self.conversation_history
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            self.logger.info(f"Conversation history saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {str(e)}", exc_info=True)
            return False

    async def cmdloop(self):
        """Main command loop."""
        self.display_welcome()
        
        while self.running:
            try:
                command = input("krod > ").strip()
                if not command:
                    continue
                await self.process_command(command)
            except KeyboardInterrupt:
                print("\nPress Ctrl+C again or type 'quit' to exit")
            except EOFError:
                print("\nThank you for using KROD. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                self.logger.error(f"CLI error: {str(e)}", exc_info=True)


def main():
    """Entry point for the CLI."""
    # Set up basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = {
        "enable_persistent_storage": True,
        "storage": {
            "type": "json",  # or "sqlite", "postgres", etc.
            "path": "~/.krod/conversations"  # default path
        },
        "llm": {
            "model": "gpt-4",  # or whatever your default model is
            "temperature": 0.7
        }
    }

    # Create and run the CLI
    cli = KrodCLI(config)

    try:
        asyncio.run(cli.cmdloop())
    except KeyboardInterrupt:
        print("\nKROD was interrupted. Goodbye!")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()