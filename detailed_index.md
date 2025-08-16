# LiveKit-Agents Detailed Technical Index

## 📂 Directory Structure

```
livekit-agents/
├── 🔧 Main Entry Points
│   ├── main.py                    # Main orchestration with prewarm() and entrypoint()
│   └── user_database.py           # UserDatabase class with SQLite operations
│
├── 🤖 Agent Core
│   └── agent/
│       ├── __init__.py            # WallmateAgent exports
│       └── wallmate_agent.py      # WallmateAgent class with full feature set
│
├── ⚙️ Configuration System
│   └── config/
│       ├── __init__.py            # Configuration exports
│       ├── base_instructions.py   # create_base_instructions() function
│       ├── language_config.py     # Language mapping functions
│       ├── session_config.py      # Session constants and defaults
│       └── voice_config.py        # VoiceSettings and ElevenLabsConfig classes
│
├── 🔗 Core Session Management
│   └── core/
│       ├── __init__.py            # SessionSetup exports
│       └── session_setup.py       # SessionSetup class with static methods
│
├── 📡 Event & Communication
│   └── handlers/
│       ├── __init__.py            # Handler exports
│       ├── event_handlers.py      # SessionEventHandlers class
│       └── rpc_handlers.py        # RPCHandlers class
│
├── 🛠️ Utilities & Extensions
│   ├── data/db_viewer.py          # Database viewing functions
│   ├── mcp/                       # MCP integration examples
│   └── styletts2/                 # Custom TTS implementation
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

---

## 🔧 Main Entry Points

### 📄 main.py
**Purpose**: Main orchestration entry point for the Wallmate Agent system

**Key Functions**:
```python
async def prewarm(proc: JobProcess) -> None:
    """Initialize VAD model with optimized threshold"""
    # Loads Silero VAD model with activation threshold for performance

async def entrypoint(ctx: JobContext) -> None:
    """Main async entrypoint orchestrating entire agent lifecycle"""
    # Complete session setup, event registration, and agent lifecycle management
```

**Key Features**:
- Agent session orchestration with complete lifecycle management
- Event handler registration for state changes, metrics, and session events
- RPC method registration for client-agent communication
- Usage metrics collection and performance monitoring
- Pre-connection audio setup for immediate response capability

**Dependencies**: `livekit.agents`, `agent.wallmate_agent`, `config.session`, `handlers`

---


### 📄 user_database.py
**Purpose**: SQLite-based user data and chat history management with security features

**Key Classes**:
```python
@dataclass
class UserData:
    """User information storage"""
    user_id: str
    name: Optional[str]
    created_at: datetime
    last_seen: datetime
    metadata: Dict[str, Any]

@dataclass  
class ChatMessage:
    """Chat message storage with metadata"""
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]

class UserDatabase:
    """Main database interface with security features"""
    
    def __init__(self, base_dir: str = "data"):
        """Initialize database with secure base directory"""
    
    def sanitize_filename(self, user_id: str) -> str:
        """Secure filename generation preventing path traversal"""
        # Prevents path traversal attacks and ensures safe filenames
    
    def get_or_create_user(self, user_id: str, metadata: Optional[Dict] = None) -> UserData:
        """User data retrieval or creation with metadata support"""
    
    def save_chat_message(self, user_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Single chat message persistence"""
    
    def save_chat_messages(self, user_id: str, messages: List[ChatMessage]) -> None:
        """Bulk chat message persistence for performance"""
    
    def get_chat_history(self, user_id: str, limit: int = 120) -> List[ChatMessage]:
        """Recent conversation retrieval with configurable limit"""
    
    def clear_chat_history(self, user_id: str) -> None:
        """Complete chat history deletion"""
    
    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """User statistics and summary information"""
```

**Security Features**:
- Path traversal prevention with filename sanitization
- SQL injection protection through parameterized queries
- Connection management with proper resource cleanup
- Secure database file creation and access control

**Database Schema**:
- **users**: user_id, name, created_at, last_seen, metadata (JSON)
- **chat_messages**: role, content, timestamp, metadata (JSON)

**Dependencies**: `sqlite3`, `json`, `datetime`, `pathlib`, `dataclasses`

---

## 🤖 Agent Core

### 📄 agent/wallmate_agent.py
**Purpose**: Core agent implementation with advanced features including face animation, multilingual support, and conversation persistence

**Key Classes**:
```python
class WallmateAgent(Agent):
    """Main production agent with comprehensive feature set"""
    
    def __init__(self, user_context: Dict[str, Any], user_language: str = "ko", 
                 custom_persona: Optional[str] = None, custom_voice: Optional[str] = None):
        """Initialize agent with user context and configuration"""
        # Sets up user context, language, persona, and voice configuration
    
    async def _prepare_chat_context_with_history(self, user_database: UserDatabase, 
                                               user_id: str) -> ChatContext:
        """Load previous conversation history into chat context"""
        # Loads up to 120 previous messages for conversation continuity
        # Prevents duplicate saves by tracking preloaded message count
    
    async def on_enter(self) -> None:
        """Generate contextual greeting for new/returning users"""
        # Provides personalized greeting based on user history and context
    
    @function_tool
    async def save_user_name(self, name: str) -> str:
        """Function tool to save user's name with system context return"""
        # Saves user name and returns system context for agent acknowledgment
```

**Key Features**:
- **Face Animation (STF)**: Speech-To-Face animation data generation
- **Multilingual Support**: Korean, English, Japanese, Chinese
- **Conversation History**: Persistent chat history with intelligent loading
- **Custom Personas**: Dynamic persona system with character maintenance
- **Voice Configuration**: ElevenLabs TTS with customizable voice presets
- **User Context Management**: Comprehensive user data and preference handling
- **Anthropic Claude Integration**: Advanced LLM capabilities with Claude models

**Architecture**:
- **Voice Pipeline**: STT → LLM (Claude) → TTS (ElevenLabs) → STF (Face Animation)
- **Session Management**: Complete lifecycle with enter/exit handling
- **Function Tools**: Extensible tool system for agent capabilities
- **Context Preservation**: Maintains conversation continuity across sessions

**Dependencies**: `livekit.agents`, `config.models`, `config.persona_config`, `user_database`

---

## ⚙️ Configuration System

### 📄 config/models.py
**Purpose**: Simple model configurations for STT, LLM, TTS, and STF components

**Key Functions**:
```python
def get_stt(language: str = "ko"):
    """Get Deepgram STT configuration"""
    return deepgram.STT(model="nova-2-general", language=language)

def get_llm():
    """Get Anthropic LLM configuration"""
    return anthropic.LLM(
        model="claude-4-sonnet-20250514",
        caching="ephemeral",
        max_tokens=256,
    )

def get_tts(voice_name: str = "FEMALE_1"):
    """Get ElevenLabs TTS configuration"""
    from config.voices import ElevenLabsConfig
    return ElevenLabsConfig.from_voice_name(voice_name).create_tts()

def get_stf():
    """Get FaceAnimator STF configuration"""
    return FaceAnimator(
        chunk_duration_sec=2.0, 
        output_mode=OutputMode.ANIMATION_ONLY
    )
```

**Key Features**:
- Simple function-based configuration without complex classes
- Direct model instantiation with optimized settings
- Easy voice configuration through voice name mapping
- Face animation support with configurable output mode
- Clear separation of concerns for each model type

**Dependencies**: `livekit.plugins.deepgram`, `livekit.plugins.anthropic`, `livekit.agents.stf`

---

### 📄 config/persona_config.py
**Purpose**: Dynamic persona-based instruction generation with system context handling

**Key Functions**:
```python
def create_instructions(user_language: str = "ko", custom_persona: Optional[str] = None) -> str:
    """Generate agent instructions with persona and language support"""
    # Creates comprehensive agent instructions with:
    # - Default "Lulu" idol persona or custom persona
    # - Language-specific communication rules
    # - Character maintenance guidelines
    # - System context handling instructions
    # - Multilingual response capabilities
```

**Default Persona**: "Lulu" - Friendly idol character with specific behavioral traits
**System Context Processing**: Handles `[SYSTEM_CONTEXT: ...]` format for dynamic responses
**Language Support**: Korean primary with multilingual fallback capabilities
**Character Guidelines**: Consistent personality maintenance across conversations

**Key Features**:
- Dynamic persona injection from metadata
- System context interpretation for guided responses
- Language-appropriate communication style
- Character consistency rules and behavioral guidelines

---

### 📄 config/language_config.py
**Purpose**: Language mapping and validation utilities for multilingual support

**Constants**:
```python
SUPPORTED_LANGUAGES = ["ko", "en", "ja", "zh"]
LANGUAGE_NAMES = {
    "ko": "한국어", "en": "English", 
    "ja": "日本語", "zh": "中文"
}
DEEPGRAM_LANGUAGE_MAPPING = {
    "ko": "ko", "en": "en-US", 
    "ja": "ja", "zh": "zh-CN"
}
DEFAULT_LANGUAGE = "ko"
```

**Key Functions**:
```python
def get_language_name(language_code: str) -> str:
    """Convert language code to display name"""

def map_language_to_deepgram(language_code: str) -> str:
    """Map internal language codes to Deepgram STT language codes"""

def is_supported_language(language_code: str) -> bool:
    """Validate if language is supported"""

def validate_language(language_code: Optional[str]) -> str:
    """Normalize and validate language codes with fallback"""
```

**Supported Languages**:
- **Korean (ko)**: Primary language with full feature support
- **English (en)**: Complete support with US regional mapping
- **Japanese (ja)**: Full support with appropriate STT mapping
- **Chinese (zh)**: Basic support with Simplified Chinese mapping

---

### 📄 config/voices.py
**Purpose**: ElevenLabs TTS voice presets and configuration management

**Key Classes**:
```python
@dataclass
class VoiceSettings:
    """Voice parameter configuration"""
    stability: float = 0.5      # Voice stability (0.0-1.0)
    similarity_boost: float = 0.5  # Similarity boost (0.0-1.0) 
    style: float = 0.0          # Style setting (0.0-1.0)
    speed: float = 1.0          # Speech speed multiplier

@dataclass
class ElevenLabsConfig:
    """Complete TTS configuration with voice presets"""
    voice_id: str
    settings: VoiceSettings
    name: str
    
    @classmethod
    def from_voice_name(cls, voice_name: str) -> 'ElevenLabsConfig':
        """Map voice names to configurations"""
        # Returns appropriate voice configuration for given name
    
    def create_tts(self) -> elevenlabs.TTS:
        """Create configured ElevenLabs TTS instance"""
        # Returns ready-to-use TTS with applied settings
```

**Voice Presets**:
- **FEMALE_1-4**: Four female voice options with unique voice IDs and optimized settings
- **MALE_1-4**: Four male voice options with distinct characteristics
- **Default Settings**: Balanced stability, similarity boost, and natural speed

**Configuration Features**:
- Voice ID mapping to ElevenLabs voice models
- Customizable voice parameters for different use cases
- Factory methods for easy TTS instance creation
- Preset management for consistent voice experiences

---

### 📄 config/session.py
**Purpose**: Simple session setup and configuration utilities

**Key Functions**:
```python
def parse_metadata(participant: rtc.RemoteParticipant) -> Dict[str, Any]:
    """Parse participant metadata with defaults"""
    # Extracts user_language, custom_persona, voice_name with fallbacks

def create_room_options(participant: rtc.RemoteParticipant):
    """Create room input and output options"""
    # Configures audio/video/animation options for optimal performance

def setup_session(participant: rtc.RemoteParticipant) -> Dict[str, Any]:
    """Complete session setup in one simple function"""
    # Orchestrates metadata parsing, user database setup, room configuration
```

**Session Setup Process**:
- Metadata parsing with safe defaults and validation
- User database initialization and language updates
- Room option configuration for animation support
- Complete session context preparation

**Key Features**:
- Single-function session setup for simplicity
- Robust error handling with graceful fallbacks
- Language validation and user preference updates
- Room configuration optimized for face animation

**Dependencies**: `livekit.rtc`, `user_database`, `config.language_config`

---


## 📡 Event & Communication Handlers

### 📄 handlers/event_handlers.py
**Purpose**: Comprehensive session event handling with metrics tracking and performance monitoring

**Key Classes**:
```python
class SessionEventHandlers:
    """Manages all session-level event handlers with comprehensive tracking"""
    
    @staticmethod
    def create_agent_state_handler(ctx: JobContext) -> Callable:
        """Handle agent state changes with RPC notifications"""
        # Monitors agent state transitions and notifies clients via RPC
        # Tracks state changes for metrics and debugging
    
    @staticmethod  
    def create_user_state_handler(ctx: JobContext, user_id: str) -> Callable:
        """Handle user state changes with inactivity timeout management"""
        # Manages user state transitions and inactivity detection
        # Implements 5-minute inactivity timeout with attention checks
    
    @staticmethod
    def create_metrics_handler(ctx: JobContext, user_id: str) -> Callable:
        """Comprehensive metrics collection and performance monitoring"""
        # Collects E2E latency metrics for complete voice pipeline
        # Tracks STT, LLM, TTS, and STF performance individually
    
    @staticmethod
    def create_session_close_handler(user_database: UserDatabase, user_id: str, 
                                   chat_ctx: ChatContext, preloaded_count: int) -> Callable:
        """Session cleanup with chat history persistence"""
        # Saves new chat messages (excluding preloaded history)
        # Performs cleanup and resource management
```

**Metrics Tracking Methods**:
```python
def _handle_eou_metrics(self, event: SpeechEvent) -> None:
    """Handle End-of-Utterance metrics with timing precision"""

def _handle_llm_metrics(self, event: LLMEvent) -> None:
    """Handle LLM response metrics and token usage tracking"""

def _handle_tts_metrics(self, event: TTSEvent) -> None:
    """Handle TTS synthesis metrics and audio generation timing"""

def _handle_stf_metrics(self, event: STFEvent) -> None:
    """Handle Speech-To-Face animation metrics and processing time"""

def _log_complete_metrics(self) -> None:
    """Log comprehensive E2E performance metrics with detailed breakdown"""
```

**Performance Monitoring**:
- **E2E Latency**: Complete voice pipeline timing from speech to response
- **Component-Level Metrics**: Individual STT, LLM, TTS, STF performance tracking
- **Resource Usage**: Memory and processing time monitoring
- **Error Tracking**: Exception monitoring and error pattern analysis
- **Session Analytics**: User engagement and interaction patterns

**Features**:
- **Inactivity Management**: 5-minute timeout with RPC attention checks
- **State Synchronization**: Real-time state sync between agent and clients
- **Chat History Persistence**: Intelligent conversation history management
- **Metrics Aggregation**: Comprehensive performance data collection

**Dependencies**: `livekit.agents.voice.events`, `user_database`, `json`, `logging`

---

### 📄 handlers/rpc_handlers.py
**Purpose**: RPC method handlers for bidirectional client-agent communication

**Key Classes**:
```python
class RPCHandlers:
    """Manages RPC method handlers for client-agent communication"""
    
    @staticmethod
    def create_interrupt_handler() -> Callable:
        """Handle agent interruption requests from clients"""
        # Allows clients to interrupt agent speech or processing
        # Provides immediate response to user interruption signals
    
    @staticmethod
    def create_attention_check_handler() -> Callable:
        """Handle attention checks for inactive users"""
        # Responds to attention check requests during user inactivity
        # Part of inactivity timeout management system
    
    @staticmethod  
    def create_send_text_input_handler(agent_session: AgentSession) -> Callable:
        """Handle direct text input from clients"""
        # Processes text messages sent directly by clients
        # Integrates text input into voice conversation flow
    
    @staticmethod
    def create_clear_chat_history_handler(user_database: UserDatabase, user_id: str,
                                        agent_session: AgentSession) -> Callable:
        """Handle chat history clearing with context reset"""
        # Clears chat history from database and resets agent context
        # Provides fresh conversation start while maintaining user data
    
    @staticmethod
    def register_all_methods(participant: LocalParticipant, user_database: UserDatabase,
                           user_id: str, agent_session: AgentSession) -> None:
        """Register all RPC methods with local participant"""
        # Registers all available RPC methods for client communication
        # Sets up complete bidirectional communication system
```

**RPC Methods**:
- **interrupt_agent**: Immediate agent interruption capability
- **check_attention**: Inactivity attention check response
- **send_text_input**: Direct text message processing
- **clear_chat_history**: Conversation reset functionality

**Communication Features**:
- **Bidirectional**: Both client-to-agent and agent-to-client communication
- **Real-time**: Immediate response to client requests
- **Context-Aware**: Integration with agent session and conversation state
- **Error Handling**: Robust error handling for network and processing issues

**Integration**: Works seamlessly with event handlers for complete session management

**Dependencies**: `livekit.agents`, `user_database`, `json`, `logging`

---

## 🛠️ Utilities & Extensions


### 📄 data/db_viewer.py
**Purpose**: Interactive SQLite database inspection tool for user data and chat history

**Key Functions**:
```python
def view_database(db_path: str) -> None:
    """Display database contents with formatted output"""
    # Shows complete database information:
    # - User metadata and statistics
    # - Complete chat history with timestamps
    # - Message count and conversation summary
    # - Formatted JSON metadata display

def list_databases(data_dir: str = "data") -> List[str]:
    """List available database files in data directory"""
    # Scans for user database files
    # Returns sorted list of available databases

def main() -> None:
    """Interactive database selection and viewing interface"""
    # Provides user-friendly database selection
    # Handles file not found and access errors
    # Supports multiple database inspection
```

**Features**:
- **User Information Display**: Complete user profile and metadata
- **Chat History Viewing**: Chronological conversation display with timestamps
- **Metadata Parsing**: JSON metadata formatting and display
- **Interactive Selection**: User-friendly database file selection
- **Error Handling**: Graceful handling of missing or corrupted databases

**Output Format**:
- User ID, name, creation date, last seen
- Message count and conversation statistics
- Complete chat history with role indicators
- Formatted JSON metadata display

**Dependencies**: `sqlite3`, `json`, `datetime`, `pathlib`

---

### 📄 mcp/server.py
**Purpose**: Simple MCP (Model Context Protocol) server demonstration

**Key Classes**:
```python
class MCPServer:
    """Basic MCP server implementation"""
    
    def __init__(self):
        """Initialize MCP server with basic configuration"""
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        # Basic request processing and response generation
    
    async def start_server(self, host: str = "localhost", port: int = 8000) -> None:
        """Start MCP server on specified host and port"""
```

**Features**:
- **Protocol Compliance**: Basic MCP protocol implementation
- **Request Handling**: Simple request/response processing
- **Configuration**: Configurable host and port settings
- **Async Support**: Asynchronous request processing

---

### 📄 mcp/mcp-agent.py
**Purpose**: Agent implementation with MCP server integration

**Key Classes**:
```python
class MCPAgent(Agent):
    """Agent with MCP server integration capabilities"""
    
    def __init__(self, mcp_server_url: str):
        """Initialize agent with MCP server connection"""
    
    async def query_mcp_server(self, query: str) -> str:
        """Query MCP server for information"""
        # Sends queries to MCP server and processes responses
    
    @function_tool
    async def mcp_lookup(self, query: str) -> str:
        """Function tool for MCP server queries"""
        # Provides MCP server access as agent tool
```

**Integration Features**:
- **MCP Protocol**: Integration with Model Context Protocol servers
- **Function Tools**: MCP queries available as agent tools
- **Error Handling**: Robust handling of server connection issues
- **Response Processing**: Intelligent processing of MCP server responses

---

### 📄 styletts2/tts.py
**Purpose**: StyleTTS2 TTS implementation for custom voice synthesis

**Key Classes**:
```python
class StyleTTS2(TTS):
    """Custom TTS implementation using StyleTTS2"""
    
    def __init__(self, model_path: str, config_path: str):
        """Initialize StyleTTS2 with model and configuration files"""
    
    async def synthesize(self, text: str) -> AudioData:
        """Synthesize speech from text using StyleTTS2"""
        # Custom voice synthesis with StyleTTS2 models
    
    def configure_voice(self, voice_settings: Dict[str, Any]) -> None:
        """Configure voice parameters for synthesis"""
        # Applies voice configuration settings
```

**Features**:
- **Custom Models**: Support for custom StyleTTS2 voice models
- **Voice Configuration**: Flexible voice parameter adjustment
- **High Quality**: Advanced neural voice synthesis
- **Integration**: Compatible with LiveKit agents TTS interface

**Dependencies**: `styletts2`, `torch`, `livekit.agents.tts`

---

## 🎯 LiveKit Framework Core

### 🔧 Core Framework

#### 📄 livekit/agents/__init__.py
**Purpose**: Main framework entry point with public API exports and component imports

**Key Exports**:
```python
# Core agent classes
from .voice import Agent, AgentSession

# Event system  
from .voice.events import *

# Exception handling
from ._exceptions import *

# Worker and job management
from .worker import Worker, WorkerOptions
from .job import JobContext, JobProcess

# Plugin system
from .plugin import Plugin

# MCP integration (dynamic import)
try:
    from .llm.mcp import *
except ImportError:
    pass
```

**Framework Components**:
- **Agent System**: Core agent and session management classes
- **Event System**: Comprehensive event handling for agent lifecycle
- **Plugin Architecture**: Extensible plugin system for AI services
- **Job Management**: Task distribution and execution framework
- **Exception Handling**: Structured error handling across framework
- **MCP Support**: Optional Model Context Protocol integration

---

#### 📄 livekit/agents/_exceptions.py
**Purpose**: Framework exception hierarchy with retry logic and API error handling

**Key Classes**:
```python
class APIError(Exception):
    """Base API error with retry capabilities"""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        """Initialize with error message and optional HTTP status code"""
    
    def should_retry(self) -> bool:
        """Determine if error is retryable based on status code"""
        # 4xx errors are generally not retryable (client errors)
        # 5xx errors are retryable (server errors)

class APIStatusError(APIError):
    """HTTP status-specific API error"""
    
class APIConnectionError(APIError):
    """Connection-related API error"""
    
class APITimeoutError(APIError):
    """Request timeout error"""

class AssignmentTimeoutError(Exception):
    """Job assignment timeout error"""
```

**Retry Logic**:
- **Automatic Retry**: 5xx status codes trigger automatic retry
- **Non-Retryable**: 4xx status codes indicate client errors
- **Connection Errors**: Network issues with retry capability
- **Timeout Handling**: Configurable timeout with retry logic

---

#### 📄 livekit/agents/types.py
**Purpose**: Core type definitions, constants, and framework-wide utility types

**Key Constants**:
```python
# Transcription attributes
ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID = "lk.transcription_segment_id"
ATTRIBUTE_TRANSCRIPTION_FINAL = "lk.transcription_final"
ATTRIBUTE_TRANSCRIPTION_LANGUAGE = "lk.transcription_language"

# Animation attributes
ATTRIBUTE_ANIMATION_OUTPUT_MODE = "lk.animation_output_mode"
ATTRIBUTE_ANIMATION_SEGMENT_ID = "lk.animation_segment_id"
ATTRIBUTE_ANIMATION_SAMPLE_RATE = "lk.animation_sample_rate"

# Topics for data streams
TOPIC_CHAT = "lk_chat"
TOPIC_TRANSCRIPTION = "lk_transcription"
TOPIC_ANIMATION_STREAM = "lk_animation_stream"
```

**Utility Types**:
```python
class NotGiven:
    """Type for optional parameters that are explicitly not provided"""
    
class APIConnectOptions:
    """Configuration for API connections with retry settings"""
    retry: int = 3
    timeout: float = 30.0
```

**Integration Points**:
- **Data Stream Attributes**: Standardized metadata for transcription and animation
- **Topic Management**: Consistent topic naming for real-time data streams
- **API Configuration**: Common connection and retry patterns

---

#### 📄 livekit/agents/job.py
**Purpose**: Job orchestration, context management, and room connection handling

**Key Classes**:
```python
class JobContext:
    """Central context for agent operations, room interactions, and API access"""
    
    def __init__(self, process: JobProcess, room: Room, agent: Agent):
        """Initialize job context with process, room, and agent"""
    
    async def connect(self, autosubscribe: bool = True) -> None:
        """Connect to WebRTC room with optional auto-subscription"""
        
    async def wait_for_participant(self, identity: str = None) -> RemoteParticipant:
        """Wait for specific participant to join the room"""
        
    def add_shutdown_callback(self, callback: Callable) -> None:
        """Register callback for graceful shutdown"""

class JobRequest:
    """Handles job acceptance/rejection with accept arguments"""
    
    def accept(self, **kwargs) -> None:
        """Accept job with optional configuration arguments"""
        
    def reject(self, reason: str = None) -> None:
        """Reject job with optional reason"""

class JobProcess:
    """Process-level configuration and userdata storage"""
    
    def __init__(self):
        """Initialize with process-specific configuration"""
        
    @property
    def userdata(self) -> Dict[str, Any]:
        """Access process-level user data storage"""
```

**Key Features**:
- **Room Management**: WebRTC room connection and participant handling
- **Lifecycle Management**: Job acceptance, execution, and cleanup
- **SIP Integration**: Support for SIP-based telephony connections
- **Process Context**: Process-level data storage and configuration
- **Shutdown Handling**: Graceful shutdown with callback system

---

#### 📄 livekit/agents/worker.py
**Purpose**: Worker process management, job assignment, and multi-executor orchestration

**Key Classes**:
```python
class Worker:
    """Main worker class with job handling and metrics"""
    
    def __init__(self, options: WorkerOptions):
        """Initialize worker with comprehensive configuration"""
    
    async def start(self) -> None:
        """Start worker with job processing loop"""
        
    async def shutdown(self) -> None:
        """Graceful shutdown with cleanup"""
        
    def set_load_threshold(self, threshold: float) -> None:
        """Configure load balancing threshold"""

class WorkerOptions:
    """Comprehensive configuration for worker behavior"""
    
    def __init__(self):
        self.entrypoint_fnc: Callable = None
        self.prewarm_fnc: Optional[Callable] = None
        self.executor_type: ExecutorType = ExecutorType.THREAD
        self.max_concurrent_jobs: int = 10
        self.memory_limit_mb: Optional[int] = None
        self.load_threshold: float = 0.8

class WorkerPermissions:
    """Room permissions and capabilities"""
    
    can_publish: bool = True
    can_subscribe: bool = True
    can_publish_data: bool = True
```

**Worker Features**:
- **Multi-Executor Support**: Thread vs process isolation options
- **Load Monitoring**: Automatic load balancing and threshold management
- **Memory Limits**: Configurable memory constraints for job execution
- **Health Monitoring**: Process health checks and monitoring
- **Telemetry Integration**: Performance metrics and usage tracking

---

### 🖥️ CLI System

#### 📄 cli/cli.py
**Purpose**: Command-line interface with start, dev, and console commands

**Key Commands**:
```python
def start_command():
    """Production deployment command"""
    # Starts worker in production mode with optimized settings
    
def dev_command():
    """Development mode with hot reload"""
    # Enables file watching and automatic restart on changes
    
def console_command():
    """Interactive console testing mode"""
    # Provides console-based testing without external dependencies
```

**Configuration**:
- **Environment Variables**: Automatic detection and configuration
- **Logging Setup**: Configurable logging levels and output
- **Worker Options**: Command-line configuration of worker behavior
- **Development Tools**: Hot reload, debugging, and testing utilities

---

### 🔄 IPC (Inter-Process Communication)

#### 📄 ipc/proc_pool.py
**Purpose**: Process pool management with supervised execution

**Key Classes**:
```python
class ProcessPool:
    """Manages pool of worker processes with health monitoring"""
    
    def __init__(self, max_processes: int = 4):
        """Initialize process pool with configurable size"""
    
    async def submit_job(self, job: Job) -> JobResult:
        """Submit job to available process in pool"""
        
    async def shutdown(self) -> None:
        """Graceful shutdown of all processes"""

class SupervisedProcess:
    """Individual process with health monitoring and restart capabilities"""
    
    def __init__(self, target: Callable, *args, **kwargs):
        """Initialize supervised process with target function"""
    
    async def start(self) -> None:
        """Start process with supervision"""
        
    async def restart(self) -> None:
        """Restart process on failure"""
```

**Features**:
- **Process Isolation**: Complete isolation between job executions
- **Health Monitoring**: Automatic process health checks and restarts
- **Load Balancing**: Intelligent job distribution across processes
- **Resource Management**: Memory and CPU monitoring per process

---

### 🧠 LLM Integration

#### 📄 llm/llm.py
**Purpose**: Core LLM abstraction with streaming, completion, and tool calling support

**Key Classes**:
```python
class LLM:
    """Base class for Large Language Model integration"""
    
    def __init__(self, model: str = None):
        """Initialize LLM with model configuration"""
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate single response from prompt"""
        
    def stream(self, prompt: str, **kwargs) -> AsyncIterator[ChatChunk]:
        """Stream response chunks for real-time processing"""
        
    async def agenerate_with_tools(self, messages: List[Message], 
                                  tools: List[FunctionTool]) -> ToolCallResult:
        """Generate response with function tool calling capability"""

class ChatChunk:
    """Streaming response chunk with content and metadata"""
    
    def __init__(self, content: str, is_final: bool = False):
        self.content = content
        self.is_final = is_final
        self.usage: Optional[CompletionUsage] = None

class CompletionUsage:
    """Token usage tracking with caching support"""
    
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None
```

**Integration Features**:
- **OpenTelemetry Tracing**: Distributed tracing for LLM calls
- **Metrics Collection**: Token usage, latency, and error tracking
- **Tool Calling**: Function tool integration with structured responses
- **Streaming Support**: Real-time response streaming for low latency
- **Provider Abstraction**: Unified interface across different LLM providers

---

#### 📄 llm/chat_context.py
**Purpose**: Conversation context management with history and tool integration

**Key Classes**:
```python
class ChatContext:
    """Manages conversation history and tool integration"""
    
    def __init__(self, messages: List[ChatMessage] = None):
        """Initialize context with optional message history"""
    
    def append(self, message: ChatMessage) -> None:
        """Add message to conversation history"""
        
    def copy(self) -> 'ChatContext':
        """Create copy of context for parallel processing"""
        
    def to_openai_ctx(self) -> List[Dict]:
        """Convert to OpenAI-compatible message format"""
        
    def to_anthropic_ctx(self) -> List[Dict]:
        """Convert to Anthropic Claude-compatible format"""

class ChatMessage:
    """Individual conversation message with role and content"""
    
    def __init__(self, role: str, content: str, tool_calls: List = None):
        self.role = role  # 'user', 'assistant', 'system', 'tool'
        self.content = content
        self.tool_calls = tool_calls or []
```

**Context Features**:
- **Message History**: Persistent conversation history management
- **Tool Integration**: Function tool calls within conversation flow
- **Context Copying**: Safe context copying for concurrent operations
- **Provider Compatibility**: Format conversion for different LLM providers
- **Read-only Contexts**: Immutable contexts for safe parallel access

---

#### 📄 llm/tool_context.py
**Purpose**: Function tool system with decorator-based tool creation

**Key Classes & Decorators**:
```python
@function_tool
def example_tool(param: str) -> str:
    """Example function tool with automatic registration"""
    return f"Result: {param}"

class FunctionTool:
    """Represents a callable function tool for LLM integration"""
    
    def __init__(self, func: Callable, description: str = None):
        """Initialize tool with function and optional description"""
    
    async def call(self, **kwargs) -> Any:
        """Execute tool with provided arguments"""
        
    def to_openai_tool(self) -> Dict:
        """Convert to OpenAI function calling format"""
        
    def to_anthropic_tool(self) -> Dict:
        """Convert to Anthropic tool format"""

class ToolContext:
    """Manages collection of tools and their execution"""
    
    def __init__(self, tools: List[FunctionTool] = None):
        """Initialize with optional tool collection"""
    
    def add_tool(self, tool: FunctionTool) -> None:
        """Add function tool to collection"""
        
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute named tool with arguments"""
```

**Tool Features**:
- **Automatic Discovery**: Decorator-based tool registration
- **Type Safety**: Automatic parameter type validation
- **Error Handling**: Robust error handling for tool execution
- **Provider Integration**: Compatible with multiple LLM providers
- **Async Support**: Full async/await support for tool execution

---

### 🎙️ Voice Processing

#### 📄 voice/agent.py
**Purpose**: Core agent logic with instruction handling and voice pipeline integration

**Key Classes**:
```python
class Agent:
    """Base agent class with tool integration and lifecycle management"""
    
    def __init__(self, instructions: str = "", tools: List[FunctionTool] = None):
        """Initialize agent with instructions and function tools"""
        self.instructions = instructions
        self.tools = tools or []
        self._tool_context = ToolContext(tools)
    
    async def on_enter(self) -> None:
        """Called when agent enters session - override for custom behavior"""
        pass
    
    async def on_exit(self) -> None:
        """Called when agent exits session - override for cleanup"""  
        pass
    
    def add_tool(self, tool: FunctionTool) -> None:
        """Add function tool to agent capabilities"""
        self.tools.append(tool)
        self._tool_context.add_tool(tool)
    
    def set_instructions(self, instructions: str) -> None:
        """Update agent instructions dynamically"""
        self.instructions = instructions
```

**Agent Features**:
- **Instruction-Based Behavior**: Flexible behavior through natural language instructions
- **Tool Integration**: Dynamic function tool system with runtime addition
- **Turn Detection**: Intelligent conversation turn management
- **Interruption Handling**: Support for user interruptions and agent control
- **Voice Pipeline Integration**: Seamless STT→LLM→TTS→STF pipeline

---

#### 📄 voice/agent_session.py
**Purpose**: Agent session management with STT→LLM→TTS→STF pipeline orchestration

**Key Classes**:
```python
class AgentSession:
    """Manages agent runtime sessions with full voice pipeline"""
    
    def __init__(self, vad: VAD, stt: STT, llm: LLM, tts: TTS, stf: STF = None):
        """Initialize session with voice pipeline components"""
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.stf = stf  # Optional face animation
    
    async def start(self, agent: Agent, room: Room) -> None:
        """Start agent session with voice pipeline"""
        
    async def stop(self) -> None:
        """Stop session and cleanup resources"""
        
    def set_chat_context(self, ctx: ChatContext) -> None:
        """Update conversation context"""
        
    async def interrupt(self) -> None:
        """Interrupt current agent processing"""

class SessionConfig:
    """Configuration for agent session behavior"""
    
    def __init__(self):
        self.turn_detection_timeout: float = 0.5
        self.interruption_enabled: bool = True
        self.vad_threshold: float = 0.7
        self.max_message_length: int = 1000
```

**Session Features**:
- **Pipeline Orchestration**: Complete STT→LLM→TTS→STF coordination  
- **VAD Integration**: Voice activity detection with configurable sensitivity
- **Activity Tracking**: Agent activity state management and monitoring
- **Real-time Processing**: Low-latency streaming throughout pipeline
- **Context Management**: Persistent conversation context across session

---

### 🎤 STT (Speech-to-Text)

#### 📄 stt/stt.py
**Purpose**: STT abstraction with streaming recognition and speaker identification

**Key Classes**:
```python
class STT:
    """Abstract base class for Speech-to-Text providers"""
    
    def __init__(self, language: str = "en-US", sample_rate: int = 16000):
        """Initialize STT with language and audio configuration"""
        self.language = language
        self.sample_rate = sample_rate
    
    def stream(self) -> "STTStream":
        """Create streaming STT session"""
        pass
    
    async def recognize(self, audio: AudioData) -> SpeechEvent:
        """Single-shot speech recognition"""
        pass

class STTStream:
    """Streaming STT interface with real-time recognition"""
    
    async def push_audio(self, audio: AudioData) -> None:
        """Push audio data for recognition"""
        pass
    
    async def flush(self) -> None:
        """Flush remaining audio and get final results"""
        pass
    
    @property
    def event_emitter(self) -> EventEmitter:
        """Access to recognition events"""
        return self._event_emitter

class SpeechEvent:
    """Speech recognition result with confidence and alternatives"""
    
    def __init__(self, transcript: str, confidence: float = 1.0):
        self.transcript = transcript
        self.confidence = confidence
        self.is_final = False
        self.alternatives: List[str] = []
        self.speaker_id: Optional[str] = None
```

**STT Features**:
- **Streaming Recognition**: Real-time speech-to-text processing
- **Speaker Identification**: Multi-speaker conversation support
- **Confidence Scoring**: Recognition confidence levels
- **Language Support**: Multi-language recognition capabilities
- **Alternative Transcripts**: Multiple recognition hypotheses

---

### 🔊 TTS (Text-to-Speech)

#### 📄 tts/tts.py
**Purpose**: TTS abstraction with streaming synthesis and voice selection

**Key Classes**:
```python
class TTS:
    """Abstract base class for Text-to-Speech providers"""
    
    def __init__(self, voice: str = None, sample_rate: int = 24000):
        """Initialize TTS with voice and audio configuration"""
        self.voice = voice
        self.sample_rate = sample_rate
    
    def stream(self) -> "TTSStream":
        """Create streaming TTS session"""
        pass
    
    async def synthesize(self, text: str) -> AudioData:
        """Single-shot text-to-speech synthesis"""
        pass

class TTSStream:
    """Streaming TTS interface with real-time synthesis"""
    
    async def push_text(self, text: str) -> None:
        """Push text for synthesis"""
        pass
    
    async def flush(self) -> None:
        """Flush remaining text and complete synthesis"""
        pass
    
    @property
    def event_emitter(self) -> EventEmitter:
        """Access to synthesis events"""
        return self._event_emitter

class TTSEvent:
    """TTS synthesis event with audio data"""
    
    def __init__(self, audio: AudioData, text: str):
        self.audio = audio
        self.text = text
        self.is_final = False
```

**TTS Features**:
- **Streaming Synthesis**: Real-time text-to-speech generation
- **Voice Selection**: Multiple voice options and characteristics
- **SSML Support**: Speech Synthesis Markup Language for advanced control
- **Audio Quality**: High-quality audio generation with configurable sample rates
- **Emotional Expression**: Voice modulation and emotional synthesis

---

### 😊 STF (Speech-to-Face)

#### 📄 stf/stf.py
**Purpose**: Face animation generation from speech with dual-mode output support

**Key Classes**:
```python
class FaceAnimator:
    """Converts audio to facial animation data"""
    
    def __init__(self, output_mode: OutputMode = OutputMode.ANIMATION_ONLY,
                 chunk_duration_sec: float = 0.5):
        """Initialize animator with output mode and timing"""
        self.output_mode = output_mode
        self.chunk_duration_sec = chunk_duration_sec
    
    def stream(self) -> "STFStream":
        """Create streaming animation session"""
        return STFStream(self)
    
    async def generate_animation(self, audio: AudioData) -> AnimationData:
        """Generate animation data from audio"""
        pass

class OutputMode(Enum):
    """Animation output modes"""
    ANIMATION_ONLY = "animation_only"           # Legacy mode
    ANIMATION_WITH_AUDIO = "animation_with_audio"  # Recommended mode

class STFStream:
    """Streaming STF interface with real-time animation generation"""
    
    async def push_audio(self, audio: AudioData) -> None:
        """Push audio data for animation generation"""
        pass
    
    async def flush(self) -> None:
        """Flush remaining audio and complete animation"""
        pass
    
    @property
    def event_emitter(self) -> EventEmitter:
        """Access to animation events"""
        return self._event_emitter
```

**STF Features**:
- **Real-time Animation**: Live facial animation generation from speech
- **Dual Mode Output**: Animation-only or animation-with-audio modes
- **Triton Integration**: High-performance inference with Triton servers
- **Synchronization**: Perfect audio-animation timing synchronization
- **WebRTC Streaming**: Direct streaming to clients via WebRTC

---

#### 📄 stf/animation_data.py
**Purpose**: Animation data structures and frame format definitions

**Key Classes**:
```python
class AnimationFrame:
    """Individual animation frame with blend shape coefficients"""
    
    def __init__(self, blend_shapes: List[float], timestamp: float):
        """Initialize frame with blend shapes and timing"""
        self.blend_shapes = blend_shapes  # 52 blend shape coefficients
        self.timestamp = timestamp
        self.duration: Optional[float] = None

class AnimationData:
    """Complete animation sequence with metadata"""
    
    def __init__(self, frames: List[AnimationFrame]):
        """Initialize with frame sequence"""
        self.frames = frames
        self.sample_rate: int = 60  # 60 FPS
        self.total_duration: float = 0.0
    
    def to_bytes(self) -> bytes:
        """Serialize animation data for transmission"""
        pass
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'AnimationData':
        """Deserialize animation data from bytes"""
        pass

class BlendShapeWeights:
    """Standard blend shape weight definitions"""
    
    # 52 standard blend shapes for facial animation
    BLEND_SHAPE_NAMES = [
        "browInnerUp", "browDownLeft", "browDownRight",
        "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft",
        # ... complete list of 52 blend shapes
    ]
```

**Animation Features**:
- **52 Blend Shapes**: Industry-standard facial animation blend shapes
- **60 FPS Animation**: Smooth animation at 60 frames per second
- **Efficient Serialization**: Optimized binary format for real-time streaming
- **Metadata Support**: Complete timing and synchronization information

---

## 🔍 Architecture Summary

The LiveKit-Agents codebase implements a sophisticated voice AI agent system with the following architectural highlights:

### 🎭 **Agent Architecture**
- **Modular Design**: Separates concerns across agent, configuration, handlers, and utilities
- **Plugin System**: Extensible integration with multiple AI service providers
- **Event-Driven**: Comprehensive event system for lifecycle and performance management
- **Tool Integration**: Flexible function tool system for agent capabilities

### 🌐 **Multilingual Support**  
- **Language Management**: Complete language configuration and validation system
- **Localization**: Korean character mapping and phonetic pronunciation
- **Voice Adaptation**: Language-specific TTS and STT configuration
- **Content Adaptation**: Dynamic persona and instruction generation

### 💾 **Data Management**
- **User Persistence**: Comprehensive user data and chat history management
- **Security**: Path traversal prevention, SQL injection protection
- **Performance**: Efficient bulk operations and connection management
- **Privacy**: Secure data handling with proper cleanup

### 📊 **Performance & Monitoring**
- **E2E Metrics**: Complete voice pipeline performance tracking
- **Component Metrics**: Individual STT, LLM, TTS, STF performance monitoring
- **Resource Tracking**: Memory, CPU, and network usage monitoring
- **Quality Assurance**: Comprehensive validation and error handling

### 🔄 **Communication Systems**
- **RPC Integration**: Bidirectional client-agent communication
- **Event Handling**: Real-time state synchronization and notification
- **Session Management**: Complete lifecycle with timeout and cleanup
- **Error Recovery**: Robust error handling and graceful degradation

This architecture supports production-ready voice AI agents with advanced features like face animation, conversation persistence, multilingual support, and comprehensive monitoring.