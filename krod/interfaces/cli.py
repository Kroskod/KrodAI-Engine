"""
KROD CLI - Command Line Interface for the KROD AI research assistant.
"""

import os
import sys
import cmd
import json
import logging
import argparse
from typing import Dict, Any, List, Optional
import readline

from ..core.engine import KrodEngine
from ..core.config import load_config

class KrodCLI(cmd.Cmd):
    """
    Interactive command-line interface for KROD.
    
    This class provides a command-line shell for interacting with KROD,
    with support for research sessions, history, and specialized commands.
    """
    
    intro = """
    ╔═════════════════════════════════════════════════════════╗
    ║                                                         ║
    ║   Krod - Knowledge-Reinforced Operational Developer     ║
    ║                                                         ║
    ║   An AI research assistant for complex problem solving  ║
    ║                                                         ║
    ║   Type 'help' for a list of commands                    ║
    ║   Type 'quit' to exit                                   ║
    ║                                                         ║
    ╚═════════════════════════════════════════════════════════╝
    """
    
    prompt = "Krod> "
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the KROD CLI.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.logger = logging.getLogger("krod.cli")
        
        # Load configuration
        self.config = config or load_config()
        
        # Initialize the KROD engine
        self.engine = KrodEngine(self.config)
        
        # Set up session state
        self.current_session_id = None
        self.last_response = None
        
        # Set up history file
        history_file = self.config.get("interfaces", {}).get("cli", {}).get("history_file", ".krod_history")
        self.history_file = os.path.expanduser(history_file)
        
        # Load history if it exists
        self._load_history()
        
        self.logger.info("KROD CLI initialized")
    
    def _load_history(self):
        """Load command history from file."""
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
                self.logger.debug(f"Loaded history from {self.history_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load history: {str(e)}")
    
    def _save_history(self):
        """Save command history to file."""
        try:
            readline.write_history_file(self.history_file)
            self.logger.debug(f"Saved history to {self.history_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save history: {str(e)}")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    def default(self, line: str):
        """
        Process input as a query to KROD if not a command.
        
        Args:
            line: The user input
        """
        if line.strip():
            self._process_query(line)
    
    def _process_query(self, query: str):
        """
        Process a query and display the response.
        
        Args:
            query: The query to process
        """
        try:
            # Process the query
            result = self.engine.process(query, self.current_session_id)
            
            # Update session ID if needed
            if "context_id" in result:
                self.current_session_id = result["context_id"]
            
            # Store the last response
            self.last_response = result
            
            # Print the response
            if "response" in result:
                print(f"\n{result['response']}\n")
            else:
                print("\nNo response from KROD.\n")
                
            # Print domain and capabilities if in debug mode
            if self.config.get("debug", False):
                if "domain" in result:
                    print(f"Domain: {result['domain']}")
                if "capabilities" in result:
                    print(f"Capabilities: {', '.join(result.get('capabilities', []))}")
                print()
                
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            print(f"\nError: {str(e)}\n")
    
    def do_quit(self, arg):
        """Exit the KROD CLI."""
        print("Goodbye! Thank you for using KROD.")
        self._save_history()
        return True
    
    def do_exit(self, arg):
        """Exit the KROD CLI."""
        return self.do_quit(arg)
    
    def do_session(self, arg):
        """
        Manage research sessions.
        
        Usage:
          session new             - Create a new session
          session list            - List available sessions
          session switch <id>     - Switch to a specific session
          session info [id]       - Show information about current or specified session
          session export <file>   - Export current session to a file
          session import <file>   - Import a session from a file
        """
        args = arg.strip().split()
        
        if not args:
            print("Current session ID:", self.current_session_id or "None")
            return
        
        cmd = args[0].lower()
        
        if cmd == "new":
            # Create a new session
            session = self.engine.research_context.create()
            self.current_session_id = session.id
            print(f"Created new session with ID: {session.id}")
            
        elif cmd == "list":
            # List available sessions
            sessions = self.engine.research_context.list_sessions()
            if not sessions:
                print("No sessions available.")
                return
                
            print("\nAvailable sessions:")
            for session in sessions:
                print(f"  ID: {session['id']}")
                print(f"  Created: {session['created_at']}")
                print(f"  Messages: {session['message_count']}")
                print()
                
        elif cmd == "switch" and len(args) > 1:
            # Switch to a specific session
            session_id = args[1]
            session = self.engine.research_context.get(session_id)
            
            if session:
                self.current_session_id = session_id
                print(f"Switched to session: {session_id}")
            else:
                print(f"Session not found: {session_id}")
                
        elif cmd == "info":
            # Show information about a session
            session_id = args[1] if len(args) > 1 else self.current_session_id
            
            if not session_id:
                print("No session selected.")
                return
                
            session = self.engine.research_context.get(session_id)
            
            if not session:
                print(f"Session not found: {session_id}")
                return
                
            print(f"\nSession ID: {session.id}")
            print(f"Created: {session.created_at}")
            print(f"Updated: {session.updated_at}")
            print(f"Messages: {len(session.history)}")
            print(f"Metadata: {json.dumps(session.metadata, indent=2)}")
            print(f"Artifacts: {len(session.artifacts)}")
            print()
            
        elif cmd == "export" and len(args) > 1:
            # Export session to a file
            if not self.current_session_id:
                print("No session selected.")
                return
                
            filepath = args[1]
            success = self.engine.research_context.save_to_file(self.current_session_id, filepath)
            
            if success:
                print(f"Session exported to: {filepath}")
            else:
                print(f"Failed to export session to: {filepath}")
                
        elif cmd == "import" and len(args) > 1:
            # Import session from a file
            filepath = args[1]
            session_id = self.engine.research_context.load_from_file(filepath)
            
            if session_id:
                self.current_session_id = session_id
                print(f"Session imported from {filepath} with ID: {session_id}")
            else:
                print(f"Failed to import session from: {filepath}")
                
        else:
            print("Unknown session command. Type 'help session' for usage.")
    
    def do_history(self, arg):
        """
        View conversation history for the current session.
        
        Usage:
          history            - Show all messages in the current session
          history <n>        - Show the last n messages
          history clear      - Clear the history display (doesn't delete the session)
        """
        if not self.current_session_id:
            print("No session selected. Use 'session new' to create one.")
            return
            
        session = self.engine.research_context.get(self.current_session_id)
        
        if not session:
            print(f"Session not found: {self.current_session_id}")
            return
            
        if arg.strip().lower() == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            return
            
        try:
            limit = int(arg) if arg.strip() else None
        except ValueError:
            print("Invalid argument. Please provide a number or 'clear'.")
            return
            
        history = session.history
        if limit:
            history = history[-limit:]
            
        if not history:
            print("No messages in this session.")
            return
            
        print("\nConversation history:")
        print("---------------------")
        
        for i, msg in enumerate(history):
            role = msg["role"].upper()
            content = msg["content"]
            timestamp = msg["timestamp"]
            
            print(f"\n[{i+1}] {role} ({timestamp}):")
            print(f"{content}")
            
        print("\n---------------------")
    
    def do_domain(self, arg):
        """
        Set or view the preferred domain for the next query.
        
        Usage:
          domain            - Show current domain preference
          domain code       - Set preferred domain to code
          domain math       - Set preferred domain to math
          domain research   - Set preferred domain to research
          domain clear      - Clear domain preference
        """
        arg = arg.strip().lower()
        
        if not arg:
            domain = self.config.get("preferred_domain")
            if domain:
                print(f"Current preferred domain: {domain}")
            else:
                print("No preferred domain set. KROD will automatically detect the domain.")
            return
            
        if arg in ["code", "math", "research"]:
            self.config["preferred_domain"] = arg
            print(f"Preferred domain set to: {arg}")
        elif arg == "clear":
            if "preferred_domain" in self.config:
                del self.config["preferred_domain"]
            print("Cleared domain preference.")
        else:
            print(f"Unknown domain: {arg}. Valid domains are: code, math, research")
    
    def do_debug(self, arg):
        """
        Toggle debug mode.
        
        Debug mode shows additional information about KROD's processing.
        """
        self.config["debug"] = not self.config.get("debug", False)
        print(f"Debug mode: {'enabled' if self.config['debug'] else 'disabled'}")
    
    def do_save(self, arg):
        """
        Save the last response to a file.
        
        Usage:
          save <filename>   - Save the last response to the specified file
        """
        if not arg:
            print("Please specify a filename.")
            return
            
        if not self.last_response:
            print("No response to save.")
            return
            
        try:
            filepath = arg.strip()
            with open(filepath, 'w') as f:
                f.write(self.last_response.get("response", ""))
            print(f"Response saved to: {filepath}")
        except Exception as e:
            print(f"Error saving response: {str(e)}")
    
    def do_config(self, arg):
        """
        View or modify configuration.
        
        Usage:
          config                    - Show all configuration
          config <key>              - Show specific configuration value
          config <key> <value>      - Set configuration value
          config save <filename>    - Save configuration to file
          config load <filename>    - Load configuration from file
        """
        args = arg.strip().split(maxsplit=1)
        
        if not args:
            # Show all configuration
            import json
            print(json.dumps(self.config, indent=2))
            return
            
        cmd = args[0].lower()
        
        if cmd == "save" and len(args) > 1:
            # Save configuration to file
            from ..core.config import save_config
            
            filepath = args[1]
            success = save_config(self.config, filepath)
            
            if success:
                print(f"Configuration saved to: {filepath}")
            else:
                print(f"Failed to save configuration to: {filepath}")
                
        elif cmd == "load" and len(args) > 1:
            # Load configuration from file
            from ..core.config import load_config
            
            filepath = args[1]
            try:
                new_config = load_config(filepath)
                self.config.update(new_config)
                print(f"Configuration loaded from: {filepath}")
            except Exception as e:
                print(f"Failed to load configuration: {str(e)}")
                
        elif len(args) == 1:
            # Show specific configuration value
            keys = cmd.split('.')
            value = self.config
            
            try:
                for key in keys:
                    value = value[key]
                
                if isinstance(value, dict):
                    print(json.dumps(value, indent=2))
                else:
                    print(value)
            except (KeyError, TypeError):
                print(f"Configuration key not found: {cmd}")
                
        elif len(args) == 2:
            # Set configuration value
            keys = cmd.split('.')
            value_str = args[1]
            
            # Try to parse the value
            try:
                # Try as JSON first
                value = json.loads(value_str)
            except json.JSONDecodeError:
                # If not valid JSON, use string as is
                value = value_str
            
            # Navigate to the right spot in the config
            config_ptr = self.config
            for key in keys[:-1]:
                if key not in config_ptr:
                    config_ptr[key] = {}
                config_ptr = config_ptr[key]
            
            # Set the value
            config_ptr[keys[-1]] = value
            print(f"Set {cmd} = {value}")
            
        else:
            print("Unknown config command. Type 'help config' for usage.")
    
    def do_about(self, arg):
        """
        Display information about KROD and its capabilities.
        
        Usage:
          about             - Show full description
          about capabilities - Show capabilities
          about features    - Show special features
          about principles  - Show guiding principles
        """
        if not hasattr(self.engine, 'identity'):
            print("Identity information not available.")
            return
        
        arg = arg.strip().lower()
        
        if arg == "capabilities":
            print(self.engine.identity.get_capabilities())
        elif arg == "features":
            print(self.engine.identity.get_features())
        elif arg == "principles":
            print(self.engine.identity.get_principles())
        else:
            print(self.engine.identity.get_full_description())
    
    def do_model(self, arg):
        """
        Display information about KROD's model and capabilities.
        
        Usage:
          model             - Show all model information
          model limitations - Show model limitations
          model ethics     - Show ethical guidelines
        """
        if not hasattr(self.engine, 'identity'):
            print("Model information not available.")
            return
        
        arg = arg.strip().lower()
        
        if arg == "limitations":
            print(self.engine.identity.get_model_info("limitations"))
        elif arg == "ethics":
            print(self.engine.identity.get_model_info("ethics"))
        else:
            print(self.engine.identity.get_model_info())


def main():
    """Run the KROD CLI."""
    parser = argparse.ArgumentParser(description="KROD - Knowledge-Reinforced Operational Developer")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--query", help="Process a single query and exit")
    parser.add_argument("--session", help="Use a specific session ID")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.debug:
        config["debug"] = True
    
    if args.query:
        # Process a single query and exit
        engine = KrodEngine(config)
        result = engine.process(args.query, args.session)
        print(result["response"])
    else:
        # Start interactive CLI
        cli = KrodCLI(config)
        if args.session:
            # Try to use the specified session
            session = cli.engine.research_context.get(args.session)
            if session:
                cli.current_session_id = args.session
                print(f"Using session: {args.session}")
            else:
                print(f"Session not found: {args.session}")
        
        try:
            cli.cmdloop()
        except KeyboardInterrupt:
            print("\nExiting KROD. Goodbye!")
        finally:
            cli._save_history()


if __name__ == "__main__":
    main()