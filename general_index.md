# LiveKit-Agents General Index

## 📂 Directory Structure

```
livekit-agents/
├── 🔧 Main Entry Points
│   ├── main.py                    # Main Wallmate Agent orchestration
│   └── user_database.py           # SQLite user data management
│
├── 🤖 Agent Core
│   └── agent/
│       ├── __init__.py            # Agent package exports
│       └── wallmate_agent.py      # Core agent with face animation
│
├── ⚙️ Configuration System
│   └── config/
│       ├── __init__.py            # Config package exports
│       ├── models.py              # Simple model functions (STT, LLM, TTS, STF)
│       ├── voices.py              # ElevenLabs voice presets
│       ├── session.py             # Simple session setup utilities
│       ├── persona_config.py      # Dynamic persona instructions
│       └── language_config.py     # Multilingual language mapping
│
├── 📡 Event & Communication
│   └── handlers/
│       ├── __init__.py            # Handler package exports
│       ├── event_handlers.py      # Session event handling
│       └── rpc_handlers.py        # Client-agent RPC methods
│
├── 🛠️ Utilities & Extensions
│   ├── data/
│   │   └── db_viewer.py           # Database inspection tool
│   ├── mcp/
│   │   ├── server.py              # MCP server demo
│   │   └── mcp-agent.py           # MCP integrated agent
│   └── styletts2/
│       ├── __init__.py            # StyleTTS2 package
│       └── tts.py                 # Custom TTS implementation
│
└── 🎯 LiveKit Framework Core
    └── livekit/agents/
        ├── 🔧 Core Framework
        │   ├── __init__.py        # Main framework entry point
        │   ├── _exceptions.py     # Framework exception hierarchy
        │   ├── types.py           # Core type definitions
        │   ├── job.py             # Job orchestration & context
        │   ├── worker.py          # Worker process management
        │   ├── plugin.py          # Plugin system infrastructure
        │   ├── inference_runner.py # Abstract inference execution
        │   ├── log.py             # Framework logging
        │   ├── vad.py             # Voice Activity Detection
        │   └── jupyter.py         # Jupyter notebook integration
        │
        ├── 🖥️ CLI System
        │   └── cli/
        │       ├── cli.py         # Command-line interface
        │       ├── _run.py        # Worker execution runtime
        │       ├── log.py         # CLI logging setup
        │       ├── proto.py       # CLI protocols
        │       └── watcher.py     # File watching for dev mode
        │
        ├── 🔄 IPC (Inter-Process Communication)
        │   └── ipc/
        │       ├── channel.py     # Communication channels
        │       ├── proc_pool.py   # Process pool management
        │       ├── job_executor.py # Job execution
        │       ├── inference_executor.py # Distributed inference
        │       └── [multiple executor & process files]
        │
        ├── 🧠 LLM Integration
        │   └── llm/
        │       ├── llm.py         # Core LLM abstraction
        │       ├── chat_context.py # Conversation context
        │       ├── tool_context.py # Function tool system
        │       ├── mcp.py         # Model Context Protocol
        │       ├── realtime.py    # Real-time LLM streaming
        │       └── _provider_format/ # Provider-specific formatting
        │           ├── anthropic.py # Claude integration
        │           ├── openai.py  # OpenAI integration
        │           └── [other providers]
        │
        ├── 🎙️ Voice Processing
        │   └── voice/
        │       ├── agent.py       # Core agent logic
        │       ├── agent_session.py # Agent session management
        │       ├── events.py      # Agent event system
        │       ├── generation.py  # Voice generation
        │       ├── avatar/        # Avatar integration
        │       ├── room_io/       # Room I/O handling
        │       ├── transcription/ # Speech transcription
        │       └── recorder_io/   # Recording capabilities
        │
        ├── 🎤 STT (Speech-to-Text)
        │   └── stt/
        │       ├── stt.py         # STT abstraction
        │       ├── stream_adapter.py # Streaming STT
        │       └── fallback_adapter.py # STT reliability
        │
        ├── 🔊 TTS (Text-to-Speech)  
        │   └── tts/
        │       ├── tts.py         # TTS abstraction
        │       ├── stream_adapter.py # Streaming TTS
        │       └── fallback_adapter.py # TTS reliability
        │
        ├── 😊 STF (Speech-to-Face)
        │   └── stf/
        │       ├── stf.py         # Face animation generation
        │       └── animation_data.py # Animation data structures
        │
        ├── 🛠️ Utilities & Support
        │   └── utils/
        │       ├── aio/           # Async utilities
        │       ├── audio.py       # Audio processing
        │       ├── codecs/        # Audio codecs
        │       ├── hw/            # Hardware monitoring
        │       ├── images/        # Image processing
        │       └── [extensive utility modules]
        │
        ├── 📊 Metrics & Monitoring
        │   ├── metrics/           # Performance tracking
        │   └── telemetry/         # OpenTelemetry integration
        │
        └── 🔤 Text Processing
            └── tokenize/          # Text tokenization
```

## 📋 File Summaries

### 🔧 Main Entry Points

| File | Purpose |
|------|---------|
| **main.py** | Main orchestration entry point for Wallmate Agent with session setup, event handling, and RPC registration |
| **user_database.py** | SQLite-based user data management with chat history, metadata, and session tracking |

### 🤖 Agent Core

| File | Purpose |
|------|---------|
| **agent/__init__.py** | Agent package exports for WallmateAgent |
| **agent/wallmate_agent.py** | Core agent implementation with multilingual support, face animation, and conversation history management |

### ⚙️ Configuration System

| File | Purpose |
|------|---------|
| **config/__init__.py** | Configuration package exports for voice, language, and session settings |
| **config/base_instructions.py** | Dynamic persona-based instruction generation with system context handling |
| **config/language_config.py** | Language mapping utilities for Korean, English, Japanese, and Chinese support |
| **config/session_config.py** | Session timeout, room configuration, and agent behavior settings |
| **config/voice_config.py** | ElevenLabs TTS voice presets and configuration management |

### 🔗 Core Session Management

| File | Purpose |
|------|---------|
| **core/__init__.py** | Core package exports for session setup utilities |
| **core/session_setup.py** | Comprehensive session initialization with metadata parsing and room configuration |

### 📡 Event & Communication Handlers

| File | Purpose |
|------|---------|
| **handlers/__init__.py** | Event and RPC handler package exports |
| **handlers/event_handlers.py** | Comprehensive session event handling with metrics tracking and chat history management |
| **handlers/rpc_handlers.py** | RPC method handlers for client-agent communication including text input and chat clearing |

### 🛠️ Utilities & Extensions

| File | Purpose |
|------|---------|
| **data/db_viewer.py** | SQLite database viewer tool for inspecting chat history and user data |
| **mcp/server.py** | Simple MCP (Model Context Protocol) server demonstration |
| **mcp/mcp-agent.py** | Agent implementation with MCP server integration |
| **styletts2/__init__.py** | StyleTTS2 package exports |
| **styletts2/tts.py** | StyleTTS2 TTS implementation for custom voice synthesis |

### 🎯 LiveKit Framework Core

#### 🔧 Core Framework

| File | Purpose |
|------|---------|
| **livekit/agents/__init__.py** | Main framework entry point with public API exports and component imports |
| **livekit/agents/_exceptions.py** | Framework exception hierarchy with retry logic and API error handling |
| **livekit/agents/types.py** | Core type definitions, constants, and framework-wide utility types |
| **livekit/agents/job.py** | Job orchestration, context management, and room connection handling |
| **livekit/agents/worker.py** | Worker process management, job assignment, and multi-executor orchestration |
| **livekit/agents/plugin.py** | Plugin system infrastructure with thread-safe registration and lifecycle |
| **livekit/agents/inference_runner.py** | Abstract inference execution system with pluggable model support |
| **livekit/agents/log.py** | Framework-wide logging configuration and development level support |
| **livekit/agents/vad.py** | Voice Activity Detection framework with streaming and metrics |
| **livekit/agents/jupyter.py** | Jupyter notebook integration with Colab support and development tools |

#### 🖥️ CLI System

| File | Purpose |
|------|---------|
| **cli/cli.py** | Command-line interface with start, dev, and console commands |
| **cli/_run.py** | Worker execution runtime with signal handling and development mode |
| **cli/log.py** | CLI-specific logging setup and configuration |
| **cli/proto.py** | CLI protocol definitions and command structures |
| **cli/watcher.py** | File watching system for development mode hot reload |

#### 🔄 IPC (Inter-Process Communication)

| File | Purpose |
|------|---------|
| **ipc/channel.py** | Communication channels between processes with async support |
| **ipc/proc_pool.py** | Process pool management with supervised execution |
| **ipc/job_executor.py** | Job execution orchestration across multiple processes |
| **ipc/inference_executor.py** | Distributed inference execution with process isolation |
| **ipc/supervised_proc.py** | Supervised process management with health monitoring |

#### 🧠 LLM Integration

| File | Purpose |
|------|---------|
| **llm/llm.py** | Core LLM abstraction with streaming, completion, and tool calling support |
| **llm/chat_context.py** | Conversation context management with history and tool integration |
| **llm/tool_context.py** | Function tool system with decorator-based tool creation |
| **llm/mcp.py** | Model Context Protocol integration for external tools and context |
| **llm/realtime.py** | Real-time LLM streaming and live conversation support |
| **llm/_provider_format/anthropic.py** | Claude/Anthropic provider-specific message formatting |
| **llm/_provider_format/openai.py** | OpenAI provider-specific message and tool formatting |

#### 🎙️ Voice Processing

| File | Purpose |
|------|---------|
| **voice/agent.py** | Core agent logic with instruction handling and voice pipeline integration |
| **voice/agent_session.py** | Agent session management with STT→LLM→TTS→STF pipeline orchestration |
| **voice/events.py** | Agent event system with lifecycle events and performance metrics |
| **voice/generation.py** | Voice generation coordination and pipeline management |
| **voice/avatar/** | Avatar integration for visual agents with animation synchronization |
| **voice/room_io/** | Room input/output handling for WebRTC audio/video streams |
| **voice/transcription/** | Speech transcription utilities with filtering and synchronization |

#### 🎤 STT (Speech-to-Text)

| File | Purpose |
|------|---------|
| **stt/stt.py** | STT abstraction with streaming recognition and speaker identification |
| **stt/stream_adapter.py** | Streaming STT adapter for real-time speech recognition |
| **stt/fallback_adapter.py** | STT reliability layer with automatic provider fallback |

#### 🔊 TTS (Text-to-Speech)

| File | Purpose |
|------|---------|
| **tts/tts.py** | TTS abstraction with streaming synthesis and voice selection |
| **tts/stream_adapter.py** | Streaming TTS adapter for real-time speech synthesis |
| **tts/fallback_adapter.py** | TTS reliability layer with automatic provider fallback |

#### 😊 STF (Speech-to-Face)

| File | Purpose |
|------|---------|
| **stf/stf.py** | Face animation generation from speech with dual-mode output support |
| **stf/animation_data.py** | Animation data structures and frame format definitions |

#### 🛠️ Utilities & Support

| File | Purpose |
|------|---------|
| **utils/aio/** | Async utilities including channels, task management, and intervals |
| **utils/audio.py** | Audio processing utilities and format handling |
| **utils/codecs/** | Audio codec support and encoder/decoder implementations |
| **utils/hw/** | Hardware monitoring including CPU usage and system resources |
| **utils/images/** | Image processing utilities for avatar and visual components |

#### 📊 Metrics & Monitoring

| File | Purpose |
|------|---------|
| **metrics/base.py** | Base metrics collection framework with component-specific tracking |
| **metrics/usage_collector.py** | Usage metrics collection for LLM, STT, TTS, and STF components |
| **telemetry/metrics.py** | OpenTelemetry metrics integration and export |
| **telemetry/traces.py** | Distributed tracing with span management and context propagation |

#### 🔤 Text Processing

| File | Purpose |
|------|---------|
| **tokenize/tokenizer.py** | Text tokenization interface for LLM input management |
| **tokenize/basic.py** | Basic tokenization implementation with word and sentence splitting |
| **tokenize/blingfire.py** | BlingFire tokenizer integration for efficient text processing |

---

## 🎭 Agent Types

- **🤖 Wallmate Agent**: Main production agent with face animation, multilingual support, and conversation history
- **🔗 MCP Agent**: Integration example with Model Context Protocol

## 🌐 Language Support

- **Korean (ko)**: Primary language with comprehensive localization support
- **English (en)**: Full support with voice presets and framework integration
- **Japanese (ja)**: Supported with proper language mapping and STT/TTS integration
- **Chinese (zh)**: Basic support with language validation and regional mapping

## 🎵 Voice & Animation

- **TTS**: ElevenLabs integration with 8 voice presets (4 female, 4 male)
- **STF**: Face animation data generation for avatar synchronization
- **Voice Settings**: Configurable stability, similarity boost, style, and speed parameters

## 📊 Key Features

### 🎤 Voice Processing Pipeline
- **STT → LLM → TTS → STF**: Complete voice-to-face animation pipeline
- **Real-time Processing**: Low-latency streaming with WebRTC integration
- **Multi-provider Support**: 35+ plugins for different AI service providers
- **Voice Activity Detection**: Advanced VAD with configurable sensitivity

### 🤖 Agent Intelligence
- **Function Tools**: Decorator-based tool system with dynamic registration
- **Context Management**: Persistent conversation history and state management
- **Custom Personas**: Dynamic persona system with character consistency
- **Multi-modal Support**: Voice, text, and face animation integration

### 🏗️ Framework Architecture
- **Plugin System**: Modular architecture with extensive provider support
- **Process Management**: Multi-executor support (thread vs process isolation)
- **Event-Driven**: Comprehensive event system for lifecycle management
- **CLI Tools**: Development, testing, and production deployment commands

### 📊 Monitoring & Reliability
- **E2E Metrics**: Complete pipeline performance tracking
- **OpenTelemetry**: Distributed tracing and metrics export
- **Fallback Systems**: Automatic provider fallback for reliability
- **Health Monitoring**: Process supervision and resource monitoring

### 💾 Data Management
- **Conversation History**: Persistent chat history with SQLite database
- **User Management**: Comprehensive user data and session tracking
- **Security Features**: Path traversal prevention and SQL injection protection
- **Session Lifecycle**: Complete lifecycle with timeout handling and cleanup