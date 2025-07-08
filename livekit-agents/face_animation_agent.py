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
from livekit.agents.stf import FaceAnimatorSTFTriton
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
    get_system_message
)

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
    def __init__(self, user_data: UserData, db: UserDatabase, user_language: str = "ko"):
        self.user_data = user_data
        self.db = db
        self.user_language = user_language
        
        # 비활성화 타이머 관련 변수
        self.last_user_activity_time = time.time()
        self.inactivity_timeout = 30.0  # 30초
        self.inactivity_task: Optional[asyncio.Task] = None
        self.is_agent_speaking = False
        
        # 언어별 대화 시작 메시지 로드
        self.conversation_starters = get_conversation_starters(self.user_language)
        
        # 성능 메트릭스 수집을 위한 변수
        self.metrics_data = {
            "stt_total_duration": 0.0,
            "llm_total_duration": 0.0,
            "tts_total_duration": 0.0,
            "stf_total_duration": 0.0,  # STF 모델 레이턴시 측정 추가
            "stt_count": 0,
            "llm_count": 0,
            "tts_count": 0,
            "stf_count": 0,
            "session_start_time": time.time(),
            "end_to_end_durations": []  # 전체 응답 시간 측정용
        }
        
        # End-to-end 레이턴시 측정을 위한 변수
        self.current_request_start_time = None
        
        # 이전 대화 컨텍스트 가져오기
        context = self.db.get_recent_context(user_data.participant_id, message_count=120)
        
        # 언어별 기본 지시사항 로드 (사용자 이름과 컨텍스트 포함)
        base_instructions = get_base_instructions(
            self.user_language, 
            user_data.display_name, 
            context
        )
            
        # STT 언어 설정 (사용자 언어에 맞춤)
        deepgram_language = map_language_to_deepgram(self.user_language)
        
        super().__init__(
            instructions=base_instructions,
            stt=deepgram.STT(model="nova-2-general", language=deepgram_language),
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
                    encoding="mp3_22050_32",
                ),
            stf=FaceAnimatorSTFTriton(chunk_duration_sec=1.0),
            # turn_detection=MultilingualModel(),
        )

    async def on_enter(self): 
        logger.info(f"FaceAgent on_enter for user: {self.user_data.participant_id}")
        
        # 활동 시간 초기화
        self.last_user_activity_time = time.time()
        
        if self.user_data.display_name:
            # 이미 이름을 아는 경우 (재방문 사용자)
            greeting = get_greeting_message(self.user_language, self.user_data.display_name, True)
            self.session.generate_reply(instructions=greeting)
        else:
            # 처음 만나는 경우 (새 사용자)
            new_user_instruction = get_greeting_message(self.user_language, None, False)
            self.session.generate_reply(instructions=new_user_instruction)
    
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
    
    def _start_inactivity_timer(self):
        """비활성화 타이머 시작 - 에이전트가 말하지 않을 때만 시작"""
        # 에이전트가 현재 말하고 있으면 타이머 시작하지 않음
        if self.is_agent_speaking:
            return
            
        if self.inactivity_task:
            self.inactivity_task.cancel()
        
        async def inactivity_check():
            try:
                await asyncio.sleep(self.inactivity_timeout)
                
                # 에이전트가 현재 말하고 있지 않은 경우에만 대화 시작
                if not self.is_agent_speaking:
                    await self._initiate_conversation()
                else:
                    logger.debug("타이머 만료 시점에 에이전트가 말하는 중이므로 대화 시작 취소")
                    
            except asyncio.CancelledError:
                logger.debug("비활성화 타이머 취소됨")
        
        logger.debug(f"{self.inactivity_timeout}초 비활성화 타이머 시작")
        self.inactivity_task = asyncio.create_task(inactivity_check())
    
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
                # 타이머 정리
                if self.inactivity_task:
                    self.inactivity_task.cancel()
            else:
                # 다른 예외는 여전히 오류로 처리
                logger.error(f"대화 시작 중 오류 발생: {e}")
    
    def _update_metrics_data(self, metrics_obj):
        """메트릭스 데이터 업데이트"""
        if isinstance(metrics_obj, metrics.STTMetrics):
            self.metrics_data["stt_total_duration"] += metrics_obj.duration
            self.metrics_data["stt_count"] += 1
            logger.debug(f"STT 메트릭스 - 지속시간: {metrics_obj.duration:.3f}초, 오디오 지속시간: {metrics_obj.audio_duration:.3f}초")
            
            # End-to-end 레이턴시 측정 시작 (사용자 입력 시작점)
            if self.current_request_start_time is None:
                self.current_request_start_time = time.time()
            
        elif isinstance(metrics_obj, metrics.LLMMetrics):
            self.metrics_data["llm_total_duration"] += metrics_obj.duration
            self.metrics_data["llm_count"] += 1
            logger.debug(f"LLM 메트릭스 - 지속시간: {metrics_obj.duration:.3f}초, TTFT: {metrics_obj.ttft:.3f}초, 토큰/초: {metrics_obj.tokens_per_second:.1f}")
            
        elif isinstance(metrics_obj, metrics.TTSMetrics):
            self.metrics_data["tts_total_duration"] += metrics_obj.duration
            self.metrics_data["tts_count"] += 1
            logger.debug(f"TTS 메트릭스 - 지속시간: {metrics_obj.duration:.3f}초, TTFB: {metrics_obj.ttfb:.3f}초, 문자 수: {metrics_obj.characters_count}")
            
            # End-to-end 레이턴시 측정 종료 (응답 완료 시점)
            if self.current_request_start_time is not None:
                end_to_end_duration = time.time() - self.current_request_start_time
                self.metrics_data["end_to_end_durations"].append(end_to_end_duration)
                logger.debug(f"End-to-end 레이턴시: {end_to_end_duration:.3f}초")
                self.current_request_start_time = None
            
        elif isinstance(metrics_obj, metrics.VADMetrics):
            logger.debug(f"VAD 메트릭스 - 추론 지속시간: {metrics_obj.inference_duration_total:.3f}초, 추론 횟수: {metrics_obj.inference_count}")
            
        # STF 모델 레이턴시 측정 (임시 구현 - 실제 STF 메트릭스가 없으므로 TTS 완료 시점에 추정)
        if isinstance(metrics_obj, metrics.TTSMetrics):
            # STF 처리 시간 추정 (실제로는 별도 메트릭스 필요)
            estimated_stf_duration = metrics_obj.audio_duration * 0.1  # 대략적인 STF 처리 시간
            self.metrics_data["stf_total_duration"] += estimated_stf_duration
            self.metrics_data["stf_count"] += 1
            logger.debug(f"STF 메트릭스 (추정) - 지속시간: {estimated_stf_duration:.3f}초")
    
    def _log_performance_summary(self):
        """성능 요약 로깅"""
        session_duration = time.time() - self.metrics_data["session_start_time"]
        
        # 평균 레이턴시 계산
        avg_stt_latency = self.metrics_data["stt_total_duration"] / max(1, self.metrics_data["stt_count"])
        avg_llm_latency = self.metrics_data["llm_total_duration"] / max(1, self.metrics_data["llm_count"])
        avg_tts_latency = self.metrics_data["tts_total_duration"] / max(1, self.metrics_data["tts_count"])
        avg_stf_latency = self.metrics_data["stf_total_duration"] / max(1, self.metrics_data["stf_count"])
        
        # End-to-end 레이턴시 통계 계산
        end_to_end_durations = self.metrics_data["end_to_end_durations"]
        avg_end_to_end = sum(end_to_end_durations) / max(1, len(end_to_end_durations))
        max_end_to_end = max(end_to_end_durations) if end_to_end_durations else 0
        min_end_to_end = min(end_to_end_durations) if end_to_end_durations else 0
        
        performance_summary = {
            "session_duration": f"{session_duration:.1f}초",
            "stt_metrics": {
                "requests": self.metrics_data["stt_count"],
                "total_duration": f"{self.metrics_data['stt_total_duration']:.3f}초",
                "avg_latency": f"{avg_stt_latency:.3f}초"
            },
            "llm_metrics": {
                "requests": self.metrics_data["llm_count"],
                "total_duration": f"{self.metrics_data['llm_total_duration']:.3f}초",
                "avg_latency": f"{avg_llm_latency:.3f}초"
            },
            "tts_metrics": {
                "requests": self.metrics_data["tts_count"],
                "total_duration": f"{self.metrics_data['tts_total_duration']:.3f}초",
                "avg_latency": f"{avg_tts_latency:.3f}초"
            },
            "stf_metrics": {
                "requests": self.metrics_data["stf_count"],
                "total_duration": f"{self.metrics_data['stf_total_duration']:.3f}초",
                "avg_latency": f"{avg_stf_latency:.3f}초"
            },
            "end_to_end_metrics": {
                "total_requests": len(end_to_end_durations),
                "avg_latency": f"{avg_end_to_end:.3f}초",
                "min_latency": f"{min_end_to_end:.3f}초",
                "max_latency": f"{max_end_to_end:.3f}초"
            }
        }
        
        logger.info(f"성능 요약: {performance_summary}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """현재 성능 통계 반환"""
        session_duration = time.time() - self.metrics_data["session_start_time"]
        
        # 평균 레이턴시 계산
        avg_stt_latency = self.metrics_data["stt_total_duration"] / max(1, self.metrics_data["stt_count"])
        avg_llm_latency = self.metrics_data["llm_total_duration"] / max(1, self.metrics_data["llm_count"])
        avg_tts_latency = self.metrics_data["tts_total_duration"] / max(1, self.metrics_data["tts_count"])
        avg_stf_latency = self.metrics_data["stf_total_duration"] / max(1, self.metrics_data["stf_count"])
        
        # End-to-end 레이턴시 통계
        end_to_end_durations = self.metrics_data["end_to_end_durations"]
        avg_end_to_end = sum(end_to_end_durations) / max(1, len(end_to_end_durations))
        
        return {
            "session_duration": session_duration,
            "stt_avg_latency": avg_stt_latency,
            "llm_avg_latency": avg_llm_latency,
            "tts_avg_latency": avg_tts_latency,
            "stf_avg_latency": avg_stf_latency,
            "end_to_end_avg_latency": avg_end_to_end,
            "total_requests": {
                "stt": self.metrics_data["stt_count"],
                "llm": self.metrics_data["llm_count"],
                "tts": self.metrics_data["tts_count"],
                "stf": self.metrics_data["stf_count"],
                "end_to_end": len(end_to_end_durations)
            }
        }
    
    def _reset_inactivity_timer(self):
        """사용자 활동 감지 시 타이머 리셋"""
        self.last_user_activity_time = time.time()
        
        if self.inactivity_task:
            self.inactivity_task.cancel()
            logger.debug("사용자 활동으로 인한 타이머 취소")
        
        # 새로운 타이머는 에이전트가 응답을 마친 후에 시작
        # (_on_agent_speech_end()에서 자동으로 시작됨)
    
    def _on_user_speech_detected(self):
        """사용자 음성 감지 시 호출"""
        logger.debug("사용자 음성 감지 - 타이머 리셋")
        self._reset_inactivity_timer()
    
    def _on_agent_speech_start(self):
        """에이전트 음성 시작 시 호출"""
        self.is_agent_speaking = True
        logger.debug("에이전트 음성 시작 - 타이머 정지")
        # 에이전트가 말하는 동안 타이머 정지
        if self.inactivity_task:
            self.inactivity_task.cancel()
    
    def _on_agent_speech_end(self):
        """에이전트 음성 종료 시 호출"""
        self.is_agent_speaking = False
        logger.debug("에이전트 음성 종료 - 타이머 시작")
        # 에이전트가 말을 끝내면 타이머 시작 (사용자 응답 대기)
        self._start_inactivity_timer()
    

def prewarm(proc: JobProcess):
    # VAD 모델 로드
    proc.userdata["vad"] = silero.VAD.load()
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
    
    # 메타데이터에서 사용자 언어 추출
    user_language = "ko"  # 기본값
    metadata = {}
    supported_languages = ["ko", "en", "ja", "zh"]  # 지원하는 언어들
    
    try:
        if participant.metadata:
            import json
            metadata = json.loads(participant.metadata)
            if "deviceLanguage" in metadata:
                detected_language = metadata["deviceLanguage"]
                if detected_language in supported_languages:
                    user_language = detected_language
                    logger.info(f"사용자 언어 감지: {user_language}")
                else:
                    logger.info(f"지원하지 않는 언어 감지: {detected_language}, 기본 언어 사용: {user_language}")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"메타데이터 파싱 오류: {e}, 기본 언어 사용: {user_language}")
    
    # 사용자별 개별 데이터베이스 생성
    db = UserDatabase(participant.identity)
    user_data = db.get_or_create_user(participant.identity)
    
    # 사용자 언어 업데이트 (새로운 언어가 감지된 경우)
    if user_data.language != user_language:
        db.update_user_language(participant.identity, user_language)
        user_data.language = user_language
        logger.info(f"사용자 언어 업데이트: {participant.identity} -> {user_language}")
    
    # 메타데이터 저장
    if metadata:
        db.update_user_metadata(participant.identity, metadata)

    # AgentSession 생성 (STF 클라이언트 포함)
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
    )
    
    # 메트릭스 수집을 위한 UsageCollector 생성
    usage_collector = metrics.UsageCollector()

    room_input_options = RoomInputOptions(
        audio_enabled=True,
        video_enabled=False,
        text_enabled=False,
        # text_enabled=False,
        participant_identity=participant.identity,
    )
    # RoomIO 옵션 설정 (애니메이션 데이터 출력 활성화)
    room_output_options = RoomOutputOptions(
        audio_enabled=False,          # 오디오 출력 비활성화 (AnimationData에 포함됨)
        transcription_enabled=True,   # 텍스트 전사 출력
        # transcription_enabled=False,   # 텍스트 전사 비활성화
        animation_enabled=True,       # 애니메이션 데이터 출력 활성화
        sync_transcription=False,
    )

    logger.info(f"애니메이션 데이터 스트리밍을 활성화했습니다. 대상: {participant.identity}")
    
    # Agent의 identity 로깅
    agent_identity = ctx.room.local_participant.identity
    logger.info(f"Agent Identity: {agent_identity}")

    # Agent 인스턴스 생성 (사용자 언어 전달)
    agent = FaceAgent(user_data, db, user_language)
    
    # 메트릭스 수집 이벤트 리스너 등록
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        # 메트릭스 로깅 (상세 정보 포함)
        metrics.log_metrics(ev.metrics)
        
        # 사용량 수집
        usage_collector.collect(ev.metrics)
        
        # 에이전트의 메트릭스 데이터 업데이트
        agent._update_metrics_data(ev.metrics)
        
        # 실시간 성능 통계 출력 (5개 요청마다)
        total_requests = agent.metrics_data["stt_count"] + agent.metrics_data["llm_count"] + agent.metrics_data["tts_count"]
        if total_requests > 0 and total_requests % 5 == 0:
            stats = agent.get_performance_stats()
            logger.info(f"실시간 성능 통계 (요청 {total_requests}개): STT평균 {stats['stt_avg_latency']:.3f}초, LLM평균 {stats['llm_avg_latency']:.3f}초, TTS평균 {stats['tts_avg_latency']:.3f}초, STF평균 {stats['stf_avg_latency']:.3f}초, E2E평균 {stats['end_to_end_avg_latency']:.3f}초")
    
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
        """사용자 상태 변경 이벤트 핸들러"""
        logger.info(f"사용자 상태 변경: {ev.old_state} -> {ev.new_state}")
        
        if ev.new_state == "speaking":
            # 사용자가 말하기 시작하면 타이머 리셋
            agent._on_user_speech_detected()
        # elif ev.new_state == "away":
        #     # 사용자가 떠나면 타이머 정지
        #     if agent.inactivity_task:
        #         agent.inactivity_task.cancel()
    
    # 에이전트 상태 변경 이벤트 리스너 등록
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        """에이전트 상태 변경 이벤트 핸들러"""
        logger.info(f"에이전트 상태 변경: {ev.old_state} -> {ev.new_state}")
        
        if ev.new_state == "speaking":
            # 에이전트가 말하기 시작 - 타이머 정지
            agent._on_agent_speech_start()
        elif ev.old_state == "speaking" and ev.new_state in ["idle", "listening"]:
            # 에이전트가 말하기 종료하고 대기 상태로 전환 - 타이머 시작
            agent._on_agent_speech_end()
        elif ev.new_state == "thinking":
            # 에이전트가 생각하는 중 - 아직 타이머 시작하지 않음
            logger.debug("에이전트가 생각하는 중 - 타이머 대기")
        elif ev.old_state == "thinking" and ev.new_state == "idle":
            # 에이전트가 생각을 끝내고 대기 상태 - 타이머 시작
            logger.debug("에이전트 생각 완료 - 타이머 시작")
            agent._on_agent_speech_end()
    
    # 세션 종료 이벤트 핸들러 - 채팅 기록 저장 및 타이머 정리
    @session.on("close")
    def on_session_close():
        """세션 종료 시 채팅 기록을 데이터베이스에 저장하고 타이머 정리"""
        # 비활성화 타이머 정리
        if agent.inactivity_task:
            agent.inactivity_task.cancel()
            logger.debug("세션 종료로 인한 비활성화 타이머 취소")
            
        # 최종 사용량 통계 로깅
        summary = usage_collector.get_summary()
        logger.info(f"세션 종료 - 최종 사용량 통계: {summary}")
        
        # 에이전트의 성능 메트릭스 요약 로깅
        agent._log_performance_summary()
        
        # 실시간 성능 모니터링 대시보드 구현 완료
        logger.info("실시간 성능 모니터링 시스템이 활성화되었습니다.")
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
    

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    ) 