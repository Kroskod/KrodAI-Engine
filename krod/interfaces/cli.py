import sys
import asyncio
import logging
import os
import json
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from rich import print
from rich.console import Console
from rich.panel import Panel
from krod.core.agent_engine import AgentEngine, AgentContext
from krod.core.llm_manager import LLMManager
from krod.core.memory.memory_manager import MemoryManager
from krod.core.config import Config

class KrodCLI:
    """
    Command Line Interface for KROD.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the CLI with required components."""
        self.config = config
        self.console = Console()
        self.running = True
        self.user_id = "cli_user"  # Could be made configurable
        self.session_id = str(uuid4())
        
        # Initialize core components
        try:
            self.llm_manager = LLMManager()
            self.memory_manager = MemoryManager()
            
            # Initialize AgentEngine with proper config
            self.engine = AgentEngine(
                llm_manager=self.llm_manager,
                memory_manager=self.memory_manager,
                config=Config(config)
            )
            
            # Initialize conversation context
            self.context = AgentContext(
                conversation_id=self.session_id,
                user_id=self.user_id,
                messages=[],
                metadata={"source": "cli"}
            )
            
            self.logger = logging.getLogger("krod.cli")
            self.logger.info("CLI initialized successfully")
            
        except Exception as e:
            self.console.print(f"[red]Error initializing KROD: {str(e)}[/red]")
            sys.exit(1)

    def display_welcome(self):
        """Display welcome message with rich formatting."""
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


    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using AgentEngine.
        
        Args:
            query: The user's input query
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Update context with user message
            self.context.messages.append({
                "role": "user",
                "content": query,
                "timestamp": str(datetime.now(timezone.utc))
            })
            
            # Process the query through AgentEngine
            response = {
                "response": "",
                "sources": [],
                "reasoning": []
            }
            
            async for update in self.engine.process_query(query, self.context, stream=True):
                if update["type"] == "response":
                    response["response"] = update["content"]
                elif update["type"] == "sources":
                    response["sources"] = update["content"]
                elif update["type"] == "reasoning":
                    response["reasoning"].extend(update["content"])
                
                # Display streaming updates
                self._display_streaming_update(update)
            
            # Update context with assistant's response
            self.context.messages.append({
                "role": "assistant",
                "content": response["response"],
                "timestamp": str(datetime.now(timezone.utc))
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def _display_streaming_update(self, update: Dict[str, Any]) -> None:
        """Display streaming updates from the agent."""
        update_type = update["type"]
        content = update["content"]
        
        if update_type == "evidence":
            self.console.print(f"\n[bold]Found {len(content)} evidence sources[/bold]")
        elif update_type == "reasoning":
            self.console.print(f"\n[bold]Reasoning:[/bold] {content[-1]}", end="\r")
        elif update_type == "reflections":
            self.console.print("\n[bold]Reflections:[/bold]")
            for ref in content:
                self.console.print(f"  • {ref}")
        elif update_type == "response":
            self.console.print("\n[bold]Response:[/bold]")
            self.console.print(Markdown(content))
        elif update_type == "error":
            self.console.print(f"[red]Error: {content}[/red]")

    def _save_history(self) -> None:
        """Save the conversation history to a file."""
        if not hasattr(self, 'conversation_history') or not self.conversation_history:
            return
            
        try:
            storage_path = self.config.get('storage', {}).get('path', '~/.krod/conversations')
            storage_dir = os.path.expanduser(storage_path)
            os.makedirs(storage_dir, exist_ok=True)
            
            # Create a filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversation_{timestamp}.json"
            filepath = os.path.join(storage_dir, filename)
            
            # Prepare data to save
            data = {
                'metadata': {
                    'version': '1.0',
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'user_id': getattr(self, 'user_id', 'anonymous'),
                    'session_id': getattr(self, 'session_id', str(uuid4()))
                },
                'conversation': self.conversation_history
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Conversation history saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {str(e)}", exc_info=True)
            raise

    async def process_command(self, command: str) -> None:
        """Process a single command."""
        if not command.strip():
            return

        self.console.print("\n[cyan]Processing your request...[/cyan]")
        
        try:
            response = ""
            sources = []
            
            # Process the query through the agent engine
            async for update in self.engine.process_query(
                query=command,
                context=self.context
            ):
                if update["type"] == "cached":
                    self.console.print("[yellow]Found in memory:[/yellow]")
                    response = update["content"]
                elif update["type"] == "evidence":
                    sources = update.get("sources", [])
                    self.console.print(f"[green]Found {len(sources)} relevant sources[/green]")
                elif update["type"] == "thinking":
                    self.console.print(f"[blue]Thinking: {update['content']}[/blue]")
                elif update["type"] == "searching":
                    self.console.print(f"[yellow]Searching: {update['content']}[/yellow]")
                elif update["type"] == "response":
                    response = update["content"]
                    sources = update.get("sources", [])
                elif update["type"] == "error":
                    self.console.print(f"[red]Error: {update['content']}[/red]")
                    self.logger.error(update["content"])
            
            # Display the final response with proper formatting
            if response:
                self.console.print("\n[green]Response:[/green]")
                self.console.print(response)
                
                if sources:
                    self.console.print("\n[cyan]Sources:[/cyan]")
                    for i, source in enumerate(sources, 1):
                        title = source.get('title', 'Untitled')
                        url = source.get('url', 'No URL')
                        self.console.print(f"  {i}. {title}")
                        if url and url != 'No URL':
                            self.console.print(f"     {url}")
            
            # Save the conversation
            self._save_conversation(command, response, sources)
            
        except Exception as e:
            self.console.print(f"[red]Error processing command: {str(e)}[/red]")
            self.logger.error(f"Error processing command: {str(e)}", exc_info=True)

    def _save_conversation(self, query: str, response: str, sources: List[Dict[str, Any]]) -> None:
        """Save the conversation to history.
        
        Args:
            query: The user's query
            response: The AI's response
            sources: List of sources used in the response
        """
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = []
            
        # Add the current exchange to history
        self.conversation_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'query': query,
            'response': response,
            'sources': sources
        })
        
        # Save to file if persistent storage is enabled
        if self.config.get('enable_persistent_storage', True):
            self._save_history()

    def show_help(self):
        """Display help information."""
        help_text = """
        [bold]Available Commands:[/bold]
        [cyan]help[/cyan]         : Show this help message
        [cyan]quit[/cyan]/[cyan]exit[/cyan]   : Exit KROD
        [cyan]vector_store[/cyan] : Manage vector store operations

        [bold]Query Examples:[/bold]
        • analyze this code: def hello(): print("Hello")
        • solve this equation: 2x + 5 = 13
        • what is the complexity of quicksort?
        • research the latest developments in quantum computing
        """
        self.console.print(Panel(help_text, title="KROD Help", border_style="blue"))

    async def cmdloop(self) -> None:
        """Main CLI loop."""
        self.display_welcome()
        
        while self.running:
            try:
                query = input("\nYou: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                    
                if query.lower() == "help":
                    self.show_help()
                    continue
                    
                # Process the query
                await self.process_command(query)
                
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to quit")
            except Exception as e:
                print(f"Error: {str(e)}")
                self.logger.error(f"CLI error: {str(e)}", exc_info=True)

    async def run(self) -> None:
        """Run the CLI in async mode."""
        try:
            # Initialize the CLI
            self.running = True
            await self.cmdloop()
            
        except Exception as e:
            print(f"Fatal error: {str(e)}")
            self.logger.error(f"Fatal error in CLI: {str(e)}", exc_info=True)
            raise

def main():
    """Entry point for the CLI."""
    # Default config - can be overridden from command line or config file
    config = {
        "enable_persistent_storage": True,
        "storage": {
            "type": "json",
            "path": "~/.krod/conversations"
        },
        "llm": {
            "model": "gpt-4",
            "temperature": 0.7
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "evidence": {
            "max_sources": 5,
            "min_confidence": 0.7,
            "use_evidence": True,
            "timeout": 30.0
        }
    }
    
    # Initialize and run the CLI
    cli = KrodCLI(config)

    try:
        import asyncio
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nKROD was interrupted. Goodbye!")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()