# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For detailed information about this LiveKit Agents codebase, please refer to the following documentation files:

- @general_index.md : Overview of the directory structure and file purposes
- @WORKFLOW.md : Step-by-step execution flow of the voice agent pipeline

These files contain comprehensive information about the codebase architecture, components, and development patterns.

## Architecture Overview

### MongoDB Memory & Token Management System
The codebase uses a sophisticated dual-purpose MongoDB system for both memory management and token tracking:

**Key Components:**
- **MongoDB Checkpointer**: Handles conversation persistence via `langgraph.checkpoint.mongodb`
- **MongoDB Store**: Manages long-term user profiles and token balances
- **Thread-Based Memory**: Each user gets a unique conversation thread (`participant_id`)
- **Tool Integration**: LangGraph tools can persist data in conversation context
- **Real-Time Token Deduction**: Character-based token deduction with live balance updates

**Configuration Files:**
- `core/mongodb_manager.py`: MongoDB connection, checkpointer, and store setup
- `core/graph_builder.py`: LangGraph integration with MongoDB checkpointer
- `core/user_profile.py`: User profile and token balance management
- `handlers/event_handlers.py`: Real-time token deduction and RPC notifications

### Agent Architecture
The main agent (`livekit-agents/main.py`) orchestrates:
1. MongoDB connection establishment and testing
2. User profile retrieval/creation with token balance loading
3. LangGraph graph creation with checkpointer and store
4. Thread-specific configuration for user persistence
5. Voice processing pipeline with token management (STT → LangGraph → TTS + Token Deduction → STF)
6. Event handling with token monitoring and MongoDB persistence

### Memory & Token Flow Pattern
```
User Input → STT → LangGraph (with MongoDB context loading) 
→ Tool Execution (with persistence) → LLM Response → 
TTS (Character Count + Token Deduction) → Token Status Check + RPC Alert → 
STF → User Output → Session End Token Persistence to MongoDB
```

## Development Patterns

### Adding New Tools
1. Define tool function with `@tool` decorator in `core/graph_builder.py`
2. Add to tools list in `create_graph_with_mongodb()`
3. Tool results automatically persist in conversation context via MongoDB checkpointer
4. Access persistent data via conversation thread state and MongoDB store

### Working with Memory & Token Management
- **Thread ID Format**: Always use `participant_id` pattern
- **State Access**: LangGraph automatically loads/saves conversation state
- **Cross-Session Continuity**: Users automatically resume previous conversations with token balance
- **Memory Testing**: Use MongoDB shell commands to inspect conversation data and user profiles
- **Token Balance**: Access via `session.userdata["token_info"]` during session
- **Token Persistence**: Automatic save at session end via `UserProfileManager.update_token_balance()`

### Token System Development
- **Adding Token Costs**: Modify `_handle_tts_metrics()` in `handlers/event_handlers.py`
- **Threshold Configuration**: Update `_token_thresholds` in `SessionEventHandlers.__init__()`
- **RPC Notifications**: Customize `_check_and_notify_token_status()` for client alerts
- **User Profile Setup**: Default 10,000 tokens granted in `UserProfileManager.get_or_create_profile()`

### Key Development Files
- **Main Entry**: `livekit-agents/main.py` - MongoDB-integrated agent orchestration with user profile loading
- **Agent Logic**: `livekit-agents/core/wallmate_agent.py` - Core agent with memory and token support  
- **Event Handling**: `livekit-agents/handlers/event_handlers.py` - Real-time token deduction and MongoDB persistence
- **User Profile**: `livekit-agents/core/user_profile.py` - Token balance and user data management
- **MongoDB Manager**: `livekit-agents/core/mongodb_manager.py` - Dual-purpose connection management (checkpointer + store)
- **Graph Builder**: `livekit-agents/core/graph_builder.py` - LangGraph setup with tools, memory, and store integration
- **RPC Handlers**: `livekit-agents/handlers/rpc_handlers.py` - Client-agent communication methods
- **Model Factory**: `livekit-agents/core/model_factory.py` - STT, TTS, STF factory functions
- **Session Manager**: `livekit-agents/core/session_manager.py` - Session setup and configuration

This architecture enables persistent, context-aware conversations with real-time token management, maintaining continuity and user balances across sessions while supporting real-time voice interactions.