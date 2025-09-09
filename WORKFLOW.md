# LiveKit Voice Agent Execution Workflow Guide

## Overview
LiveKit Voice Agent is a complex pipeline that processes real-time voice conversations. This document explains the complete execution flow from user voice input to agent response, step by step.

## Core Components
- **AgentSession**: Manages overall session and WebRTC connections
- **Agent**: Handles custom logic and hook processing
- **AgentActivity**: Manages the lifecycle of a single conversation turn
- **AudioRecognition**: Voice recognition through VAD and STT
- **VAD Model**: Voice Activity Detection
- **STT Model**: Speech-to-Text conversion
- **LLM Model**: Large Language Model inference
- **TTS Model**: Text-to-Speech conversion
- **TTS Stream Pacer**: Lazy TTS inference with intelligent buffering
- **Function Tools**: External functions that the LLM can invoke

## Step-by-Step Execution Flow

### Step 1: Session Initialization and Agent Startup
```
1. AgentSession creates Room I/O and establishes WebRTC connection
2. Agent.on_enter() method is called
3. Agent speaks initial greeting (e.g., "Hello!")
4. AgentActivity instance is created and AudioRecognition is initialized
5. Audio/video forwarding tasks are started
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

### Step 5: LLM Inference and Function Tools Execution
```
1. AgentActivity._pipeline_reply_task() starts
2. perform_llm_inference() is called
3. Custom logic is applied through Agent.llm_node()
4. Streaming inference request is sent to LLM model

LLM Streaming Response Processing Loop:
- Text chunk received → Forward to text stream
- Function Call request received → Call perform_tool_executions()
  - Execute requested functions in parallel
  - Add execution results to LLM context
  - LLM continues inference based on function results

5. LLM inference completes
```

### Step 6: TTS Generation and Audio Output
```
1. perform_tts_inference() is called
2. Custom logic is applied through Agent.tts_node()
3. LLM text stream is input to TTS model

Lazy TTS Inference (with Stream Pacer):
- Text is buffered in SentenceStreamPacer
- Monitors remaining audio duration (default: 5 seconds)
- Only sends text to TTS when audio buffer is running low
- Reduces waste from interruptions by not generating unused audio

TTS Streaming Generation Loop:
- TTS model generates audio frames with aligned transcription text
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