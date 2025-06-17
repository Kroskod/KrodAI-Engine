import sys
import asyncio
import logging
from typing import Dict, Any
from rich import print
from rich.console import Console
from rich.panel import Panel

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
        self.current_session_id = None
        self.history = []
        self.running = True
        
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

    def process_command(self, command: str) -> bool:
        """Process a command or query."""
        command = command.strip()
        
        if not command:
            return True
            
        # Handle built-in commands
        if command.lower() == 'quit' or command.lower() == 'exit':
            self.running = False
            print("\nThank you for using KROD. Goodbye!")
            return False
            
        elif command.lower() == 'help':
            self.show_help()
            return True

        elif command.startswith('vector_store'):
            try:
                args = command.split()[1:]
                # Create a standalone context and invoke the command
                ctx = vector_store_commands.make_context('vector_store', args)
                vector_store_commands.invoke(ctx)
                return True
            except click.ClickException as e:
                e.show()
                return True
            except Exception as e:
                print(f"Error executing vector store command: {str(e)}", file=sys.stderr)
                return True


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

    def cmdloop(self):
        """Main command loop."""
        self.display_welcome()
        
        while self.running:
            try:
                command = input("krod> ")
                self.process_command(command)
            except KeyboardInterrupt:
                print("\nPress Ctrl+C again or type 'quit' to exit")
            except EOFError:
                print("\nThank you for using KROD. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                self.logger.error(f"CLI error: {str(e)}", exc_info=True)

    def _save_history(self):
        """Save command history."""
        # Implement history saving if needed
        pass