# LiveKit Voice Agent Execution Workflow Guide

## Overview
LiveKit Voice Agent is a complex pipeline that processes real-time voice conversations. This document explains the complete execution flow from user voice input to agent response, step by step.

## Core Components
- **AgentSession**: Manages overall session and WebRTC connections
- **Agent**: Handles custom logic and hook processing with LangGraph integration
- **AgentActivity**: Manages the lifecycle of a single conversation turn
- **AudioRecognition**: Voice recognition through VAD and STT
- **VAD Model**: Voice Activity Detection
- **STT Model**: Speech-to-Text conversion
- **LLM Model**: Large Language Model inference via LangGraph with MongoDB checkpointer
- **MongoDB Checkpointer**: Persistent conversation memory and state management
- **LangGraph Tools**: Function tools with persistent execution context
- **TTS Model**: Text-to-Speech conversion
- **TTS Stream Pacer**: Lazy TTS inference with intelligent buffering

## Step-by-Step Execution Flow

### Step 1: Session Initialization and Agent Startup
```
1. MongoDB connection is established and tested
2. User profile is retrieved/created from MongoDB Store with token balance (default: 10,000 tokens)
3. LangGraph with MongoDB checkpointer is initialized for the user
4. Thread-specific configuration is created (thread_id: "user_{participant_id}")
5. AgentSession creates Room I/O and establishes WebRTC connection with token info in userdata
6. Agent.on_enter() method is called with persistent memory context
7. Agent speaks initial greeting (e.g., "Hello!")
8. AgentActivity instance is created and AudioRecognition is initialized
9. Audio/video forwarding tasks are started
```

### Step 2: User Voice Input Processing
```
1. User provides voice input through microphone
2. Audio frames are received via WebRTC
3. Room I/O forwards to input.audio stream
4. AgentSession calls AgentActivity.push_audio()
5. Voice data is passed to AudioRecognition.push_audio()
```

### Step 3: Parallel VAD and STT Processing
**VAD (Voice Activity Detection) Processing:**
```
1. AudioRecognition._vad_task() starts
2. VAD model analyzes audio frames and returns voice activity probability (0.0~1.0)
3. When speech start is detected, AgentActivity.on_start_of_speech() is called
4. If agent is currently speaking, _maybe_interrupt() handles interruption
```

**STT (Speech-to-Text) Processing:**
```
1. AudioRecognition._stt_task() starts
2. STT model converts speech to text
3. Interim transcription results → AgentActivity.on_interim_transcript()
4. Final transcription results → AgentActivity.on_final_transcript()
```

### Step 4: Turn Detection and Response Generation Initiation
```
1. VAD detects END_OF_SPEECH event
2. AudioRecognition._run_eou_detection() is executed

Turn Detection Mode Processing:
- VAD Mode: Wait for min_endpointing_delay before determining turn end
- STT Mode: Wait for END_OF_SPEECH signal from STT
- Realtime LLM Mode: Turn detection handled by LLM server

3. AgentActivity.on_end_of_turn() is called
4. Agent.on_user_turn_completed() hook is executed
5. AgentActivity._generate_reply() starts
```

### Step 5: LLM Inference with LangGraph and MongoDB Memory
```
1. AgentActivity._pipeline_reply_task() starts
2. perform_llm_inference() is called
3. Custom logic is applied through Agent.llm_node()
4. LangGraph processes the request with MongoDB checkpointer context

LangGraph Processing with Persistent Memory:
- Current conversation state is loaded from MongoDB checkpointer
- User message is added to the persistent conversation thread
- LLM processes message with full conversation history
- Tool calls are detected and routed to ToolNode
- Tools execute with persistent context (e.g., save_user_name_langgraph)
- Tool results are added to conversation and persisted
- Final response is generated and conversation state is saved to MongoDB

5. LLM inference completes with automatic state persistence
```

### Step 6: TTS Generation, Token Deduction, and Audio Output
```
1. perform_tts_inference() is called
2. Custom logic is applied through Agent.tts_node()
3. LLM text stream is input to TTS model

Lazy TTS Inference (with Stream Pacer):
- Text is buffered in SentenceStreamPacer
- Monitors remaining audio duration (default: 5 seconds)
- Only sends text to TTS when audio buffer is running low
- Reduces waste from interruptions by not generating unused audio

TTS Streaming Generation Loop with Token Deduction:
- TTS model generates audio frames with aligned transcription text
- TTS metrics are captured including character count
- Real-time token deduction: characters_used = tokens_to_deduct (1:1 ratio)
- Token balance updated in session.userdata["token_info"]
- Token status checked against thresholds (Normal >500, Low ≤500, Critical ≤200, Depleted 0)
- RPC notification sent to client if token status worsens
- Forward to Room I/O through perform_audio_forwarding()
- Output audio stream and transcription text simultaneously

4. Agent voice is delivered to user via WebRTC
5. Transition to waiting state for next user input
```

### Step 7: Interruption Handling (Optional)
```
When user interrupts while agent is speaking:
1. New voice input is detected through VAD
2. AgentActivity.on_start_of_speech() is called
3. Current ongoing TTS is immediately stopped with interrupt()
4. New user input processing begins (return to Step 3)
```

### Step 8: Preemptive Generation - Performance Optimization
```
When preemptive_generation is enabled:
1. LLM inference starts with interim transcription while user is still speaking
2. AgentActivity.on_preemptive_generation() is called
3. LLM inference is prepared with partial context
4. When turn actually ends, output immediately with pre-prepared response
5. Overall response latency is reduced from 300ms → 100ms

Caveat: If user changes their statement or provides additional information,
discard prepared response and regenerate with new context
```

### Step 9: Session Closure and Token Persistence
```
When session ends (user disconnect, timeout, or error):
1. Session close event is triggered with reason (USER_LEFT, USER_INACTIVITY, etc.)
2. Final token balance is retrieved from session.userdata["token_info"]
3. UserProfileManager.update_token_balance() saves complete token state to MongoDB Store
4. Final token statistics are logged (remaining, total_used, total_granted)
5. RPC notification sent to client about session closure
6. MongoDB checkpointer automatically persists conversation state
7. Session cleanup and resource deallocation
8. Room connection is terminated

Token Persistence Details:
- Token balance is maintained in memory during session for performance
- Only final state is written to MongoDB to minimize database writes
- Cross-session continuity: Next session loads saved token balance
- Token usage statistics are preserved for analytics and billing
```

## Key Design Features

### Asynchronous Parallel Processing
- VAD and STT run simultaneously to minimize latency
- Function Tools execute in parallel for improved performance
- All audio processing operates in non-blocking manner

### Streaming Pipeline
- LLM responses are forwarded to TTS immediately upon generation
- TTS audio is also streamed to user as soon as it's generated
- Pipeline structure minimizes overall latency

### Lazy TTS Inference (Stream Pacer)
- Intelligent text buffering before TTS generation
- Maintains minimum audio buffer (configurable, default 5 seconds)
- Maximum text chunk size (default 300 characters)
- Significantly reduces TTS waste during interruptions
- Improves speech quality by providing more context to TTS

### Adaptive Turn Detection
- Multiple modes (VAD/STT/Realtime LLM) for context-appropriate turn detection
- Optimized according to user speaking patterns and network conditions

### Interruption Handling Mechanism
- Immediate interruption detection through real-time VAD
- Safe interruption of all ongoing generation tasks
- Works efficiently with lazy TTS to minimize resource waste
- Essential feature for natural conversation flow

This workflow systematically manages the complexity of real-time voice conversations to provide a natural and highly responsive AI agent experience.

## MongoDB Memory & Token Management Architecture

### Dual-Purpose MongoDB System
The system uses a sophisticated dual-layer architecture built on MongoDB for both memory management and token tracking:

```
┌─────────────────────────────────────────────────────────────┐
│              MongoDB Dual-Purpose System                   │
├─────────────────────────────────────────────────────────────┤
│  Short-term Memory (MongoDB Checkpointer)                  │
│  - Conversation threads per user                           │
│  - Message history with timestamps                         │
│  - Tool execution results                                  │
│  - Session state and context                              │
│                                                           │
│  Long-term Store (MongoDB Store)                          │
│  - User profiles with token balances                      │
│  - Token usage tracking (total_granted, total_used)       │
│  - Cross-session user preferences                         │
│  - Persistent user data and relationship tracking         │
└─────────────────────────────────────────────────────────────┘
```

### Thread & Token Management
- **Thread ID Format**: `user_{participant_id}` for unique user identification
- **Automatic Persistence**: All messages and tool results are automatically saved
- **Session Continuity**: Users can resume conversations with token balance across sessions
- **Context Loading**: Previous conversation history and token balance loaded on session start
- **Token Balance**: Default 10,000 tokens allocated to new users

### Real-Time Token Processing
- **Character-Based Billing**: 1 TTS character = 1 token deduction
- **Session Memory**: Token balance maintained in `session.userdata["token_info"]` for performance
- **Real-Time Alerts**: Client notifications when token status changes (Normal→Low→Critical→Depleted)
- **Persistence Strategy**: Final token state saved to MongoDB Store at session end only

### Tool Integration with Memory & Tokens
- **Persistent Tool Context**: Tools like `save_user_name_langgraph` store results in conversation memory
- **Cross-Tool Communication**: Tool results are available to subsequent tool calls
- **Memory-Aware Responses**: LLM responses consider both immediate context and persistent memory
- **Token-Aware Processing**: System monitors token usage to prevent service interruption

This architecture ensures that the agent maintains context and learns from user interactions while providing consistent, personalized responses and accurate token tracking across sessions.