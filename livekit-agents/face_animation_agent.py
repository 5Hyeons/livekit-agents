import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import time
import random

# Set higher logging level for Numba before other configurations
logging.getLogger('numba').setLevel(logging.WARNING)

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
    utils,
    RunContext,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero, elevenlabs
# from livekit.plugins.turn_detector.multilingual import MultilingualModel


# STF 모듈 임포트 및 기본 URL 정의
from livekit.agents.stf import FaceAnimator, OutputMode
from livekit.agents.voice.agent import Agent
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions
from livekit.agents.voice import MetricsCollectedEvent

# 사용자 데이터베이스 임포트
from user_database import UserDatabase, UserData, ChatMessage

# 언어 로더 임포트
from language_loader import (
    get_base_instructions,
    get_conversation_starters, 
    get_greeting_message,
    get_system_message,
    get_rpc_message
)

# 오디오 로깅 유틸리티 임포트
from livekit.agents.voice.audio_logger import log_audio_frame_info

load_dotenv()  # .env 파일에서 환경 변수 로드
logger = logging.getLogger("face-animation-agent")


def get_language_name(language_code: str) -> str:
    """언어 코드를 언어 이름으로 변환 (지원 언어: 한국어/영어/일본어/중국어)"""
    language_mapping = {
        "ko": "한국어",
        "en": "English", 
        "ja": "日本語",
        "zh": "中文",
    }
    return language_mapping.get(language_code, "한국어")  # 지원하지 않는 언어는 기본값으로 한국어

def map_language_to_deepgram(language_code: str) -> str:
    """언어 코드를 Deepgram STT 언어 코드로 매핑 (지원 언어: 한국어/영어/일본어/중국어)"""
    deepgram_mapping = {
        "ko": "ko",
        "en": "en", 
        "ja": "ja",
        "zh": "zh",
    }
    
    # 지원하지 않는 언어는 기본값으로 한국어
    return deepgram_mapping.get(language_code, "ko")


class FaceAgent(Agent):
    def __init__(self, user_data: UserData, db: UserDatabase, user_language: str = "ko", custom_persona: str = ""):
        self.user_data = user_data
        self.db = db
        self.user_language = user_language
        self.custom_persona = custom_persona
        
        # 언어별 대화 시작 메시지 로드
        self.conversation_starters = get_conversation_starters(self.user_language)
        
        # 반응성 메트릭스 추적 (스트리밍 기반)
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from streaming_reactivity_tracker import StreamingReactivityTracker
        self.reactivity_tracker = StreamingReactivityTracker()
        
        # 이전 대화 컨텍스트 가져오기
        context = self.db.get_recent_context(user_data.participant_id, message_count=120)
        
        # 언어별 기본 지시사항 로드 (사용자 이름, 컨텍스트, 커스텀 페르소나 포함)
        base_instructions = get_base_instructions(
            self.user_language, 
            user_data.display_name, 
            context,
            custom_persona=self.custom_persona
        )
            
        super().__init__(
            instructions=base_instructions,
            stt=deepgram.STT(model="nova-2-general", language=self.user_language),
            # stt=openai.STT(model="gpt-4o-transcribe", language=self.user_language),
            llm=openai.LLM(model="gpt-4o"),
            tts=elevenlabs.TTS(
                    voice_id="tZarJVdIxWQ9lIXIV9qg",
                    model="eleven_turbo_v2_5",
                    voice_settings=elevenlabs.VoiceSettings(
                        stability=0.5,
                        similarity_boost=0.75,
                        style=0.0,
                        speed=1.0,
                    ),
                    encoding="mp3_44100_32",
                ),
            stf=FaceAnimator(chunk_duration_sec=0.5, output_mode=OutputMode.ANIMATION_WITH_AUDIO),
            # turn_detection=MultilingualModel(),
        )

    async def on_enter(self): 
        logger.info(f"FaceAgent on_enter for user: {self.user_data.participant_id}")
        
        if self.user_data.display_name:
            # 이미 이름을 아는 경우 (재방문 사용자)
            greeting = get_greeting_message(self.user_language, self.user_data.display_name, True)
            await self.session.say(text=greeting, allow_interruptions=False)
            logger.info(f"재방문 사용자 인사 메시지 전송: {greeting}")
        else:
            # 처음 만나는 경우 (새 사용자)
            new_user_greeting = get_greeting_message(self.user_language, None, False)
            await self.session.say(text=new_user_greeting, allow_interruptions=False)
            logger.info(f"새 사용자 인사 메시지 전송: {new_user_greeting}")
    
    @function_tool
    async def save_user_name(self, name: str):
        """
        Call this function when the user tells you their name.
        Args:
            name: The user's name
        """
        
        # 이미 이름이 저장되어 있다면 중복 호출 방지
        if self.user_data.display_name and self.user_data.display_name == name:
            logger.info(f"Name already saved: {name}")
            return
        
        self.db.update_user_name(self.user_data.participant_id, name)
        self.user_data.display_name = name
        logger.info(f"User name saved: {self.user_data.participant_id} -> {name}")
        result = get_system_message(self.user_language, "name_saved", name=name)
        return result
    
    async def _initiate_conversation(self):
        """비활성화 시 대화 시작"""
        try:
            # 세션 유효성 검사
            if not hasattr(self, 'session') or self.session is None:
                logger.debug("세션이 없어 대화 시작 취소")
                return
                
            # 랜덤하게 대화 시작 메시지 선택
            starter_message = random.choice(self.conversation_starters)
            
            logger.info(f"비활성화 감지 - 대화 시작: {starter_message}")
            
            # 에이전트가 먼저 대화 시작
            await self.session.say(text=starter_message)
            
            # 타이머는 에이전트가 말을 끝낸 후 자동으로 시작됨 (중복 방지)
            
        except Exception as e:
            error_msg = str(e)
            if "no activity context found" in error_msg or "agent is not running" in error_msg:
                # 세션이 종료된 정상적인 상황
                logger.debug(f"세션 종료로 인한 대화 시작 취소: {error_msg}")
            else:
                # 다른 예외는 여전히 오류로 처리
                logger.error(f"대화 시작 중 오류 발생: {e}")
    
    def _update_metrics_data(self, metrics_obj):
        """반응성 메트릭스 업데이트 - 스트리밍 기반 추적"""
        # 모든 메트릭스 객체를 StreamingReactivityTracker에 전달
        # 스트리밍 파이프라인에 최적화된 TTFT/TTFB 측정
        self.reactivity_tracker.record_metrics(metrics_obj)
    
    def _log_performance_summary(self):
        """반응성 요약 로깅 (세션 종료 시)"""
        current_metrics = self.reactivity_tracker.get_current_metrics()
        if any(v is not None for v in current_metrics.values()):
            logger.info(f"Final reactivity metrics: {current_metrics}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """현재 반응성 통계 반환"""
        return self.reactivity_tracker.get_current_metrics()

def prewarm(proc: JobProcess):
    # VAD 모델 로드
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.6)
    # 데이터베이스는 각 사용자별로 개별 생성하므로 prewarm에서 제거

async def entrypoint(ctx: JobContext):
    logger.info(f"{ctx.room.name} 방에 연결합니다")
    # 오디오만 구독 (STT 용)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 첫 번째 참가자가 연결될 때까지 대기
    participant = await ctx.wait_for_participant()
    logger.info(f"{participant.identity} 참가자를 위한 음성-얼굴 에이전트 시작")
    
    # 초기 메타데이터 로깅
    logger.info(f"참가자 초기 메타데이터: {participant.metadata}")
    
    # 메타데이터에서 사용자 언어 및 커스텀 설정 추출
    user_language = "ko"  # 기본값
    agent_language = "ko"  # 기본값 (현재는 user_language와 동일)
    custom_persona = ""    # 기본값 (빈 문자열 - 기본 페르소나 사용)
    metadata = {}
    supported_languages = ["ko", "en", "ja", "zh"]  # 지원하는 언어들
    
    try:
        if participant.metadata:
            import json
            metadata = json.loads(participant.metadata)
            
            # 새로운 메타데이터 구조 확인
            if "userLanguage" in metadata:
                detected_language = metadata["userLanguage"]
                if detected_language in supported_languages:
                    user_language = detected_language
                    logger.info(f"사용자 언어 감지 (userLanguage): {user_language}")
                else:
                    logger.info(f"지원하지 않는 사용자 언어: {detected_language}, 기본 언어 사용: {user_language}")
            
            # Agent 언어 확인 (현재는 user_language와 동일하게 처리)
            if "agentLanguage" in metadata:
                detected_agent_language = metadata["agentLanguage"]
                if detected_agent_language in supported_languages:
                    agent_language = detected_agent_language
                    logger.info(f"Agent 언어 감지 (agentLanguage): {agent_language}")
                else:
                    logger.info(f"지원하지 않는 Agent 언어: {detected_agent_language}, 사용자 언어와 동일하게 설정: {user_language}")
                    agent_language = user_language
            else:
                agent_language = user_language
            
            # 커스텀 페르소나 확인
            if "customPersona" in metadata:
                custom_persona = metadata["customPersona"]
                if custom_persona and custom_persona.strip():
                    logger.info(f"커스텀 페르소나 감지 (길이: {len(custom_persona)})")
                else:
                    logger.info("빈 커스텀 페르소나 - 기본 페르소나 사용")
                    custom_persona = ""
            
            # 하위 호환성: 기존 deviceLanguage도 지원
            elif "deviceLanguage" in metadata and "userLanguage" not in metadata:
                detected_language = metadata["deviceLanguage"]
                if detected_language in supported_languages:
                    user_language = detected_language
                    agent_language = detected_language
                    logger.info(f"사용자 언어 감지 (deviceLanguage - 호환성): {user_language}")
                else:
                    logger.info(f"지원하지 않는 언어 감지: {detected_language}, 기본 언어 사용: {user_language}")
                    
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"메타데이터 파싱 오류: {e}, 기본값 사용: user_language={user_language}, agent_language={agent_language}")
    
    # 사용자별 개별 데이터베이스 생성
    db = UserDatabase(participant.identity)
    user_data = db.get_or_create_user(participant.identity)
    
    # 사용자 언어 업데이트 (새로운 언어가 감지된 경우)
    if user_data.language != user_language:
        db.update_user_language(participant.identity, user_language)
        user_data.language = user_language
        logger.info(f"사용자 언어 업데이트: {participant.identity} -> {user_language}")
    
    # 언어 및 페르소나 설정 로깅
    logger.info(f"최종 설정 - 사용자 언어: {user_language}, Agent 언어: {agent_language}, 커스텀 페르소나: {'설정됨' if custom_persona else '기본값'}")
    
    # 메타데이터 저장
    if metadata:
        db.update_user_metadata(participant.identity, metadata)

    # AgentSession 생성 (STF 클라이언트 포함)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
    )
    
    # 메트릭스 수집을 위한 UsageCollector 생성
    usage_collector = metrics.UsageCollector()
    
    # Agent 상태 변화 추적을 위한 이벤트 리스너
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        """에이전트 상태 변경 이벤트 핸들러 - speaking 시작 감지"""
        if ev.new_state == "speaking":
            # 에이전트가 말하기 시작할 때 기록 (올바른 타이밍)
            agent.reactivity_tracker.record_agent_utterance_start()
        logger.info(f"에이전트 상태 변경: {ev.old_state} -> {ev.new_state}")
        
        # Client에 agent state 변경 알림 RPC 호출
        try:
            import json
            payload = json.dumps({
                "old_state": ev.old_state,
                "new_state": ev.new_state,
                "timestamp": time.time()
            })
            
            # 비동기 작업을 동기 핸들러에서 실행
            task = asyncio.create_task(
                ctx.room.local_participant.perform_rpc(
                    destination_identity=participant.identity,  # 연결된 참가자에게 전송
                    method="agent_state_changed",
                    payload=payload,
                    response_timeout=1.0  # 1초 타임아웃
                )
            )
            
            # 태스크 완료 콜백 추가 (에러 로깅용)
            def handle_rpc_result(future):
                try:
                    future.result()
                    logger.debug(f"Agent state RPC sent successfully: {ev.new_state}")
                except Exception as e:
                    logger.warning(f"Failed to send agent state RPC: {e}")
            
            task.add_done_callback(handle_rpc_result)
            
        except Exception as e:
            logger.error(f"Error sending agent state RPC: {e}")

    room_input_options = RoomInputOptions(
        audio_enabled=True,
        video_enabled=False,
        text_enabled=False,
        # text_enabled=False,
        participant_identity=participant.identity,
    )
    # RoomIO 옵션 설정 (애니메이션 데이터 출력 활성화)
    room_output_options = RoomOutputOptions(
        audio_enabled=False,           # 오디오 출력 활성화 (ANIMATION_ONLY 모드 테스트)
        transcription_enabled=False,   # 텍스트 전사 출력
        # transcription_enabled=False,   # 텍스트 전사 비활성화
        animation_enabled=True,        # 애니메이션 데이터 출력 활성화 (ANIMATION_ONLY 모드)
        sync_transcription=False,
    )

    logger.info(f"애니메이션 데이터 스트리밍을 활성화했습니다. 대상: {participant.identity}")
    
    # Agent의 identity 로깅
    agent_identity = ctx.room.local_participant.identity
    logger.info(f"Agent Identity: {agent_identity}")

    # Agent 인스턴스 생성 (사용자 언어 및 커스텀 페르소나 전달)
    agent = FaceAgent(user_data, db, user_language, custom_persona)
    
    # 메트릭스 수집 이벤트 리스너 등록
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        # 메트릭스 로깅 (상세 정보 포함)
        metrics.log_metrics(ev.metrics)
        
        # 사용량 수집
        usage_collector.collect(ev.metrics)
        
        # 에이전트의 메트릭스 데이터 업데이트
        agent._update_metrics_data(ev.metrics)
        
    
    # 세션 시작
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=room_input_options,
        room_output_options=room_output_options
    )
    
    # 사용자 상태 변경 이벤트 리스너 등록
    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        """사용자 상태 변경 이벤트 핸들러 - 사용자 발화 종료 추적"""
        if ev.old_state == "speaking" and ev.new_state == "listening":
            # 사용자가 말을 끝낼 때 - 반응속도 측정 시작점
            agent.reactivity_tracker.record_user_speech_end()
        logger.info(f"사용자 상태 변경: {ev.old_state} -> {ev.new_state}")
        
    
    # 세션 종료 이벤트 핸들러 - 채팅 기록 저장 및 타이머 정리
    @session.on("close")
    def on_session_close():
        """세션 종료 시 채팅 기록을 데이터베이스에 저장하고 타이머 정리"""
        
        # 최종 사용량 통계 로깅
        summary = usage_collector.get_summary()
        logger.info(f"세션 종료 - 최종 사용량 통계: {summary}")
        
        # 에이전트의 반응성 메트릭스 요약 로깅
        agent._log_performance_summary()
        # agent.chat_ctx에서 현재 세션의 메시지들 가져오기
        chat_messages = []
        for item in agent.chat_ctx.items:
            if isinstance(item, llm.ChatMessage):
                if item.role in ["system", "developer"]:
                    continue
                # timestamp 변환 (float인 경우 datetime으로 변환)
                if isinstance(item.created_at, (int, float)):
                    timestamp = datetime.fromtimestamp(item.created_at)
                elif isinstance(item.created_at, datetime):
                    timestamp = item.created_at
                else:
                    timestamp = datetime.now()
                
                # content 변환 (리스트인 경우 텍스트만 추출)
                content_str = ""
                if isinstance(item.content, list):
                    content_str = " ".join(str(c) for c in item.content)
                else:
                    content_str = str(item.content)
                
                chat_messages.append(ChatMessage(
                    participant_id=participant.identity,
                    session_id=user_data.session_id,
                    timestamp=timestamp,
                    role=item.role,
                    content=content_str,
                    interrupted=getattr(item, 'interrupted', False)
                ))
        
        logger.info(f"세션 종료, 채팅 기록 저장 중... (총 {len(chat_messages)}개 메시지)")
        
        # 데이터베이스에 채팅 기록 저장
        if chat_messages:
            db.save_chat_messages(chat_messages)
            db.update_last_seen(participant.identity)
            
        # 사용자 요약 정보 로깅
        user_summary = db.get_user_summary(participant.identity)
        logger.info(f"사용자 정보: {user_summary}")
    
    # RPC 메서드 등록 - 에이전트 중단
    @ctx.room.local_participant.register_rpc_method("interrupt_agent")
    async def interrupt_agent(data: rtc.RpcInvocationData) -> None:
        """클라이언트에서 에이전트를 중단시키는 RPC 메서드"""
        logger.info(f"RPC 'interrupt_agent' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            # 현재 진행 중인 에이전트 활동 중단
            await session.interrupt()
            logger.info("AgentSession interrupt 호출 완료")
        except Exception as e:
            logger.error(f"에이전트 중단 처리 중 오류: {e}")
    
    logger.info("RPC 메서드 'interrupt_agent' 등록 완료")
    
    # RPC 메서드 등록 - 관심 확인 (1시간+ 무반응)
    @ctx.room.local_participant.register_rpc_method("check_attention")
    async def check_attention(data: rtc.RpcInvocationData) -> None:
        """1시간 이상 무반응 시 관심 확인 RPC 메서드"""
        logger.info(f"RPC 'check_attention' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            message = get_rpc_message(user_language, "check_attention")
            await session.say(text=message)
            logger.info(f"관심 확인 메시지 전송 완료: {message}")
        except Exception as e:
            logger.error(f"관심 확인 메시지 처리 중 오류: {e}")
    
    # RPC 메서드 등록 - 아침 인사 (6-8시)
    @ctx.room.local_participant.register_rpc_method("morning_greeting")
    async def morning_greeting(data: rtc.RpcInvocationData) -> None:
        """아침 시간대 인사 RPC 메서드"""
        logger.info(f"RPC 'morning_greeting' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            message = get_rpc_message(user_language, "morning_greeting")
            await session.say(text=message)
            logger.info(f"아침 인사 메시지 전송 완료: {message}")
        except Exception as e:
            logger.error(f"아침 인사 메시지 처리 중 오류: {e}")
    
    # RPC 메서드 등록 - 오전 응원 (8-10시)
    @ctx.room.local_participant.register_rpc_method("morning_boost")
    async def morning_boost(data: rtc.RpcInvocationData) -> None:
        """오전 시간대 응원 RPC 메서드"""
        logger.info(f"RPC 'morning_boost' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            message = get_rpc_message(user_language, "morning_boost")
            await session.say(text=message)
            logger.info(f"오전 응원 메시지 전송 완료: {message}")
        except Exception as e:
            logger.error(f"오전 응원 메시지 처리 중 오류: {e}")
    
    # RPC 메서드 등록 - 점심 시간 (12-14시)
    @ctx.room.local_participant.register_rpc_method("lunch_time")
    async def lunch_time(data: rtc.RpcInvocationData) -> None:
        """점심 시간 대화 RPC 메서드"""
        logger.info(f"RPC 'lunch_time' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            message = get_rpc_message(user_language, "lunch_time")
            await session.say(text=message)
            logger.info(f"점심 시간 메시지 전송 완료: {message}")
        except Exception as e:
            logger.error(f"점심 시간 메시지 처리 중 오류: {e}")
    
    # RPC 메서드 등록 - 오후 스트레칭 (16-18시)
    @ctx.room.local_participant.register_rpc_method("afternoon_stretch")
    async def afternoon_stretch(data: rtc.RpcInvocationData) -> None:
        """오후 스트레칭 제안 RPC 메서드"""
        logger.info(f"RPC 'afternoon_stretch' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            message = get_rpc_message(user_language, "afternoon_stretch")
            await session.say(text=message)
            logger.info(f"오후 스트레칭 메시지 전송 완료: {message}")
        except Exception as e:
            logger.error(f"오후 스트레칭 메시지 처리 중 오류: {e}")
    
    # RPC 메서드 등록 - 저녁 대화 (20-22시)
    @ctx.room.local_participant.register_rpc_method("evening_chat")
    async def evening_chat(data: rtc.RpcInvocationData) -> None:
        """저녁 시간 대화 RPC 메서드"""
        logger.info(f"RPC 'evening_chat' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            message = get_rpc_message(user_language, "evening_chat")
            await session.say(text=message)
            logger.info(f"저녁 대화 메시지 전송 완료: {message}")
        except Exception as e:
            logger.error(f"저녁 대화 메시지 처리 중 오류: {e}")
    
    # RPC 메서드 등록 - 늦은 밤 케어 (0-2시)
    @ctx.room.local_participant.register_rpc_method("late_night_care")
    async def late_night_care(data: rtc.RpcInvocationData) -> None:
        """늦은 밤 케어 RPC 메서드"""
        logger.info(f"RPC 'late_night_care' 호출됨! 호출자: {data.caller_identity}")
        
        try:
            message = get_rpc_message(user_language, "late_night_care")
            await session.say(text=message)
            logger.info(f"늦은 밤 케어 메시지 전송 완료: {message}")
        except Exception as e:
            logger.error(f"늦은 밤 케어 메시지 처리 중 오류: {e}")
    
    logger.info("모든 RPC 메서드 등록 완료: check_attention, morning_greeting, morning_boost, lunch_time, afternoon_stretch, evening_chat, late_night_care")
    

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    ) 