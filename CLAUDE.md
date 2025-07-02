# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **LiveKit Agents** framework - a Python framework for building realtime voice AI agents that can see, hear, and speak. The repository provides a comprehensive ecosystem for creating server-side agentic applications with WebRTC integration.

## Architecture

### Core Framework (`livekit-agents/`)
- **Agent & AgentSession**: Core agent logic and session management for user interactions
- **Voice Pipeline**: Complete voice processing chain (STT → LLM → TTS) with VAD integration
- **Plugin System**: Modular architecture supporting multiple AI service providers
- **Job Scheduling**: Built-in task distribution system with dispatch APIs
- **WebRTC Integration**: Real-time audio/video communication via LiveKit server

### Plugin Ecosystem (`livekit-plugins/`)
Extensive plugin system with 35+ integrations:
- **STT**: Deepgram, AssemblyAI, Azure, OpenAI, Google, etc.
- **LLM**: OpenAI, Anthropic, Google, AWS Bedrock, Groq, etc.
- **TTS**: ElevenLabs, OpenAI, Cartesia, Azure, Google, etc.
- **Avatars**: Tavus, Hedra, Bithuman, Bey for video avatar integration
- **Specialized**: Turn detection, VAD (Silero), MCP integration

### Development Structure
- **Workspace Setup**: UV-based monorepo with workspace members
- **Examples**: Comprehensive examples in `examples/` covering voice agents, avatars, primitives
- **Testing**: Comprehensive test suite with Docker-based integration testing

## Development Commands

### Environment Setup
```bash
# Install with basic plugins
pip install "livekit-agents[openai,silero,deepgram,cartesia,turn-detector]~=1.0"

# Development dependencies (uses UV)
uv sync --all-extras --dev
```

### Running Agents
```bash
# Terminal testing (no external dependencies)
python myagent.py console

# Development with hot reloading
python myagent.py dev

# Production deployment
python myagent.py start
```

### Quality Assurance
```bash
# Linting and formatting
ruff check
ruff format

# Type checking
mypy

# Running tests
pytest
pytest tests/test_specific.py  # Single test file

# Docker-based integration tests
cd tests/
make test PLUGIN=plugin_name
```

## Agent Development Patterns

### Basic Agent Structure
All agents follow this pattern:
```python
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    agent = Agent(instructions="...")
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=provider.STT(),
        llm=provider.LLM(),
        tts=provider.TTS(),
    )
    
    await session.start(agent=agent, room=ctx.room)
```

### Entry Points
- Use `cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))` pattern
- All examples in `examples/` directory follow this pattern
- Agent files typically end with `if __name__ == "__main__": cli.run_app(...)`

### Multi-Agent Systems
- Support agent handoff between different specialized agents
- Use `@function_tool` for tool creation and agent transitions
- Session context and userdata management for state persistence

## Key Configuration Files

- **Root `pyproject.toml`**: Workspace configuration, dev dependencies, linting rules
- **`livekit-agents/pyproject.toml`**: Main package configuration with extensive optional dependencies
- **Plugin `pyproject.toml`**: Individual plugin configurations
- **`tests/Makefile`**: Docker-based testing infrastructure
- **`uv.lock`**: Dependency lock file for reproducible builds

## Environment Variables

Core variables for LiveKit integration:
- `LIVEKIT_URL`: LiveKit server URL
- `LIVEKIT_API_KEY`: API key for authentication  
- `LIVEKIT_API_SECRET`: API secret for authentication

Provider-specific API keys (examples):
- `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, etc.

## Testing Strategy

- **Unit Tests**: Core functionality testing
- **Integration Tests**: Docker-based with toxiproxy for network simulation
- **Plugin Testing**: Individual plugin validation
- **Example Validation**: All examples serve as integration tests

## Common Development Tasks

### Adding New Plugins
1. Create plugin directory in `livekit-plugins/`
2. Follow existing plugin structure (STT/TTS/LLM base classes)
3. Add to workspace members in root `pyproject.toml`
4. Add optional dependency in main `pyproject.toml`

### Creating Agents
1. Start from `examples/voice_agents/basic_agent.py`
2. Customize Agent instructions and tools
3. Select appropriate plugin combinations
4. Test with `console` mode first, then `dev` mode

### Plugin Integration
- Import from `livekit.plugins.provider_name`
- Use consistent initialization patterns across providers
- Handle API credentials via environment variables