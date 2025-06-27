import asyncio
from krod.core.agent_engine import AgentEngine, AgentContext, AgentConfig
from krod.core.llm_manager import LLMManager
from krod.core.memory.memory_manager import MemoryManager

async def main():
    # Initialize dependencies (use your real or mock implementations as needed)
    llm_manager = LLMManager()
    memory_manager = MemoryManager()

    # Optionally customize config
    config = AgentConfig(
        max_evidence_sources=3,
        min_evidence_confidence=0.5,
        enable_reflection=True,
        enable_streaming=True,
        research_agent={},
        reasoning={}
    )

    # Initialize the AgentEngine
    agent = AgentEngine(
        llm_manager=llm_manager,
        memory_manager=memory_manager,
        config=vars(config)  # or just config, depending on your constructor
    )

    # Create a fresh context for the test user/session
    context = AgentContext(
        conversation_id="test_convo_001",
        user_id="test_user_001"
    )

    # Test query
    query = "What is quantum computing and why is it important?"

    print("==== AgentEngine Test Run ====")
    async for update in agent.process_query(query, context, stream=True):
        print(f"[{update['type'].upper()}]: {update['content']}\n")

if __name__ == "__main__":
    asyncio.run(main())