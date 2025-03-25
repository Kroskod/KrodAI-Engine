import sys
import logging
from typing import Dict, Any
from rich import print
from rich.console import Console
from rich.panel import Panel

from krod.core.engine import KrodEngine

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
    ║   KROD - Knowledge-Reinforced Operational Developer     ║
    ║                                                         ║
    ║   Version: 0.1.0                                        ║
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
            
        # Process as query
        try:
            result = self.engine.process(command, self.current_session_id)
            
            # Display response in a nice panel
            self.console.print(Panel(
                result["response"],
                title="KROD Response",
                border_style="blue"
            ))
            
            # Show token usage if available
            if "token_usage" in result:
                print(f"\nToken usage: {result['token_usage']}")
            
            # Add to history
            self.history.append({"query": command, "response": result})
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            self.logger.error(f"Query error: {str(e)}", exc_info=True)
        
        return True

    def show_help(self):
        """Show help information."""
        help_text = """
        Available Commands:
        ------------------
        help         : Show this help message
        quit/exit   : Exit KROD
        
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