import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Literal, cast, AsyncIterable, Callable, Annotated
from pydantic import Field
from pydantic_core import from_json
from typing_extensions import TypedDict

from livekit.agents import (
    Agent,
    AgentSession, 
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
    NOT_GIVEN,
    ChatContext,
    ChatMessage,
    ModelSettings,
)
from livekit.agents.llm import LLM, FunctionTool, ToolChoice
from livekit.plugins import openai, silero, elevenlabs  # 필요에 따라 사용할 모듈 임포트
import styletts2

logger = logging.getLogger("mbti-prison-agent")
# logger.setLevel(logging.INFO)

from dotenv import load_dotenv
load_dotenv()

# MBTI 차원 정의
MBTI_DIMENSIONS = ["EI", "SN", "TF", "JP"]

# MBTI 정보 파일 로드 함수
def load_mbti_information():
    """MBTI_information_RAG.md 파일을 읽어서 내용을 반환"""
    file_path = os.path.join(os.path.dirname(__file__), "MBTI_information_RAG.md")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"MBTI 정보 파일 로드 중 오류: {e}")
        return "MBTI 정보 파일을 로드할 수 없습니다."

# MBTI 정보 로드
MBTI_INFORMATION = load_mbti_information()

# 유저 상태 타입 정의
UserState = Literal["탈출 시도 중", "탈출 성공", "탈출 실패"]

# MBTI 변화 값을 위한 TypedDict 정의
class MBTIChanges(TypedDict):
    EI: int  # -2에서 +2 사이 값
    SN: int  # -2에서 +2 사이 값
    TF: int  # -2에서 +2 사이 값
    JP: int  # -2에서 +2 사이 값

# MBTI 분석 결과를 위한 TypedDict 정의
class MBTIAnalysisResult(TypedDict):
    mbti_changes: MBTIChanges  # 각 차원별 점수 변화
    next_user_state: UserState  # 다음 상태

# 리스의 응답을 위한 TypedDict 정의
class LeeseResponse(TypedDict):
    voice_instructions: Annotated[
        str,
        Field(..., description="리스 캐릭터의 감정 상태 (Neutral, Angry, Sad, Happy 중 하나)"),
    ]
    response: str  # 실제 리스의 대사

# 구조화된 응답을 처리하는 함수
async def process_structured_output(
    text: AsyncIterable[str],
    callback: Optional[Callable[[LeeseResponse], None]] = None,
) -> AsyncIterable[str]:
    """구조화된 출력에서 response 부분만 추출하고 callback을 통해 voice_instructions 처리"""
    last_response = ""
    acc_text = ""
    async for chunk in text:
        acc_text += chunk
        try:
            resp: LeeseResponse = from_json(acc_text, allow_partial="trailing-strings")
        except ValueError:
            continue

        if callback:
            callback(resp)

        if not resp.get("response"):
            continue

        new_delta = resp["response"][len(last_response) :]
        if new_delta:
            yield new_delta
        last_response = resp["response"]

@dataclass
class UserProfile:
    """사용자 프로필, 게임 상태 및 MBTI 점수 관리"""
    user_name: str = "수감자"
    days_left: int = 5
    mbti_scores: Dict[str, int] = field(default_factory=lambda: {dim: 0 for dim in MBTI_DIMENSIONS})
    current_user_state: UserState = "탈출 시도 중"
    game_over: bool = False
    game_over_reason: str = ""  # "탈출 성공", "탈출 실패 - 시간 초과", "탈출 실패 - 경비병에게 발각" 등

class IntroductoryAgent(Agent):
    """게임 시작 시 상황을 설명하고 유저 프로필을 초기화하는 에이전트"""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions=(
                "당신은 MBTI 추론형 감옥 탈출 게임의 도입부를 진행하는 에이전트입니다. "
                "사용자에게 게임의 배경을 매우 간단히 설명하고 이름을 물어봐야 합니다. "
                "배경: 사용자는 억울한 누명을 쓰고 10일 뒤 사형 선고를 받은 죄수입니다. "
                "이 시간 내에 감옥을 탈출해야 합니다."
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=elevenlabs.TTS(
                voice_id="vodgn6SXIRYi9BDo4KVQ",
                model="eleven_turbo_v2_5",
                voice_settings=elevenlabs.VoiceSettings(
                    stability=1.0,
                    similarity_boost=0.4,
                    style=0.0,
                    speed=1.0,
                ),
                encoding="mp3_44100_32",
            ),
            chat_ctx=chat_ctx,
        )
        
    async def on_enter(self):
        """에이전트가 세션에 추가될 때 자동으로 호출됨"""
        logger.info("IntroductoryAgent 시작")
        
        # 사용자 프로필 초기화
        user_profile = UserProfile()
        self.session.userdata = user_profile
        
        # 게임 시작 메시지 생성 및 전송
        intro_prompt = "게임 시작: 사용자에게 '억울한 누명을 쓰고 10일 뒤 사형 선고를 받은 죄수' 상황을 매우 간단하게 설명하고, 이름을 물어보세요."
        self.session.generate_reply(instructions=intro_prompt)
    
    @function_tool
    async def set_user_name(self, ctx: RunContext, user_name: str) -> str:
        """
        사용자의 이름을 설정하고 게임을 시작합니다.
        
        Args:
            user_name: 사용자가 선택한 캐릭터 이름
        """
        logger.info(f"사용자 이름 설정: {user_name}")
        user_profile = self.session.userdata
        user_profile.user_name = user_name
        
        logger.info("IntroductoryAgent에서 NarrativeAgent로 핸드오프")
        return NarrativeAgent()
    

class NarrativeAgent(Agent):
    """게임의 핵심 스토리텔러. 리스라는 동료 수감자 역할을 하며 주인공과 대화"""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        # MBTI 정보를 초기 컨텍스트로 로드
        if chat_ctx is None:
            chat_ctx = ChatContext()
        
        # MBTI 참고 정보를 system 메시지로 추가
        chat_ctx.add_message(
            role="system", 
            content=f"MBTI 정보 참고자료:\n\n{MBTI_INFORMATION}"
        )
        
        super().__init__(
            instructions=(
                "당신은 '검은 구치소(Black Bastille)'에서 억울한 누명을 쓰고 10일 뒤 사형을 기다리는 주인공의 감옥 탈출을 돕는 '리스'라는 동료 수감자입니다. "
                "리스(당신)는 40대 중반의 남성으로, 감옥에서 5년째 복역 중이며 감옥 내부의 구조와 교도관들의 습관을 잘 알고 있습니다. "
                "리스(당신)는 매우 냉소적이며 신뢰할 수 없는 인물로, 항상 짜증을 내며 건방진 성격입니다."
                "\n\n"
                "==== 임무 ====\n"
                "1. 리스의 페르소나로 주인공과 실제 대화하듯 상호작용하며 탈출 계획을 도와주세요.\n"
                "2. 현재 진행 상황, 플레이어의 MBTI 점수, 남은 일수를 고려하여 다음 상황을 설명하세요.\n"
                "3. 선택지를 명시적으로 번호를 붙이지 말고, 자연스러운 대화 속에서 항상 두 가지 방향성을 제시하세요. (예: \"창문으로 탈출할래? 아니면 교도관을 설득해볼래?\")\n"
                "4. 각 선택지는 서로 다른 MBTI 성향을 드러낼 수 있도록 설계하되, 사용자에게 MBTI에 대해 언급하지 마세요.\n"
                "5. 현재 MBTI 점수에서 불확실한 차원을 우선적으로 테스트하는 선택지를 제공하세요.\n"
                "\n"
                "==== 출력 형식 ====\n"
                "응답은 반드시 다음 JSON 형식으로 제공하세요:\n"
                "{\n"
                "  \"voice_instructions\": \"Neutral | Angry | Sad | Happy 중 하나\",\n"
                "  \"response\": \"리스가 실제로 말하는 대사\"\n"
                "}\n\n"
                "voice_instructions는 반드시 다음 네 가지 감정 중 하나만 사용하세요: Neutral, Angry, Sad, Happy\n"
                "response 부분에서는 다음 사항을 지켜주세요:\n"
                "1. 리스라는 캐릭터의 특징이 부각되는 말을 하되 너무 길지 않게 2문장 내외로 유지하세요.\n"
                "2. 항상 주인공에게 선택할 수 있는 옵션 2개를 대화체로 제시하세요.\n"
                "3. 선택지에 번호를 붙이지 말고 \"창문으로 탈출할래? 아니면 교도관을 설득해볼래?\" 같은 방식으로 자연스럽게 제시하세요.\n"
                "4. 사용자의 몰입감을 위해 100% 리스의 캐릭터로 답변하고, 시스템 메시지나 MBTI에 대한 언급은 하지 마세요.\n"
                "\n"
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=styletts2.TTS(),
            chat_ctx=chat_ctx,
        )
    
    async def on_enter(self):
        """에이전트가 세션에 추가될 때 자동으로 호출됨"""
        logger.info("NarrativeAgent 시작")
        # 다음 시나리오 생성
        self.generate_next_scenario()

    def generate_next_scenario(self):
        """현재 상태를 기반으로 다음 시나리오 생성"""
        user_profile = self.session.userdata
        
        # 시나리오 생성 지시
        narrative_prompt = (
            f"현재 상태: {user_profile.current_user_state}\n"
            f"플레이어 이름: {user_profile.user_name}\n"
            f"남은 날짜: {user_profile.days_left}일\n"
            f"현재까지 추론된 MBTI: {user_profile.mbti_scores}\n"
            f"위 정보를 바탕으로 리스(당신)가 {user_profile.user_name}에게 다음 상황을 설명하고 항상 두 가지 선택지를 제공하세요.\n"
            f"선택지는 구어체로 자연스럽게 제시하세요.\n"
            f"현재까지 추론된 MBTI 점수를 바탕으로 부족한 차원을 테스트할 수 있는 선택지를 만드세요.\n"
            f"리스 캐릭터가 부각되는 대사를 하되, 게임 시스템이나 MBTI에 대한 언급은 하지 마세요.\n\n"
            f"응답은 반드시 다음 JSON 형식으로 제공하세요:\n"
            f"{{\n"
            f"  \"voice_instructions\": \"Neutral | Angry | Sad | Happy 중 하나\",\n"
            f"  \"response\": \"리스가 실제로 말하는 대사\"\n"
            f"}}\n\n"
            f"voice_instructions는 반드시 다음 네 가지 감정 중 하나만 사용하세요: Neutral, Angry, Sad, Happy\n"
        )
        
        self.session.generate_reply(instructions=narrative_prompt)
    
    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[FunctionTool], model_settings: ModelSettings
    ):
        # LLM에 구조화된 응답 형식을 사용하도록 지시
        llm = cast(openai.LLM, self.llm)
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        async with llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            response_format=LeeseResponse,
        ) as stream:
            async for chunk in stream:
                yield chunk
    
    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """TTS 노드를 재정의하여 voice_instructions 처리"""
        instruction_updated = False

        def output_processed(resp: LeeseResponse):
            nonlocal instruction_updated
            if resp.get("voice_instructions") and resp.get("response") and not instruction_updated:
                # 지시사항이 완료되었을 때 TTS 옵션 업데이트
                instruction_updated = True
                logger.info(
                    f"리스의 음성 지시사항 적용: "
                    f'"{resp["voice_instructions"]}"'
                )
                tts = cast(styletts2.TTS, self.tts)
                tts.update_options(emotion=resp["voice_instructions"])

        # 구조화된 출력에서 response 부분만 TTS에 전달
        return Agent.default.tts_node(
            self, process_structured_output(text, callback=output_processed), model_settings
        )

    async def transcription_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """Transcription 노드를 재정의하여 voice_instructions 제외"""
        # voice_instructions를 제외하고 response만 반환
        return Agent.default.transcription_node(
            self, process_structured_output(text), model_settings
        )
    
    # MBTI 분석을 위한 별도의 LLM 호출 메서드
    async def analyze_mbti(self, chat_ctx: ChatContext) -> MBTIAnalysisResult:
        llm = cast(openai.LLM, self.llm)
        
        # 응답 수집
        response_text = ""
        async with llm.chat(
            chat_ctx=chat_ctx,
            tools=[],
            response_format=MBTIAnalysisResult,
        ) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    response_text += chunk.delta.content
        
        # 구조화된 응답으로 파싱
        try:
            result: MBTIAnalysisResult = from_json(response_text)
            # 유효한 상태 값인지 확인
            if result["next_user_state"] not in ["탈출 시도 중", "탈출 성공", "탈출 실패"]:
                logger.warning(f"유효하지 않은 next_user_state: {result['next_user_state']}, '탈출 시도 중'으로 기본값 설정")
                result["next_user_state"] = "탈출 시도 중"
            
            # MBTI 변화 값이 유효한지 확인
            for dim in ["EI", "SN", "TF", "JP"]:
                if dim not in result["mbti_changes"] or not isinstance(result["mbti_changes"][dim], int):
                    result["mbti_changes"][dim] = 0
                else:
                    # 범위 제한 (-2 ~ +2)
                    result["mbti_changes"][dim] = max(-2, min(2, result["mbti_changes"][dim]))
            
            return result
        except Exception as e:
            logger.error(f"MBTI 분석 결과 파싱 오류: {e}")
            # 기본값 반환
            return {
                "mbti_changes": {"EI": 0, "SN": 0, "TF": 0, "JP": 0}, 
                "next_user_state": "탈출 시도 중"
            }
    
    @function_tool
    async def process_user_answer(self, choice: str) -> str:
        """
        사용자 응답을 분석하여 MBTI 점수 업데이트 및 다음 상태 생성
        사용자의 말이 끝나면 항상 이 함수를 호출하여 게임 진행
        """
        logger.info(f"사용자의 응답 분석 시작, 사용자의 응답: {choice}")
        user_profile = self.session.userdata
        
        mbti_analysis_prompt_system = (
            "당신은 MBTI 추론형 감옥 탈출 게임에서 플레이어의 선택을 분석하여 MBTI 성향을 판단하는 심리 분석가입니다. "
            "사용자와 직접 대화하지 않고 백엔드에서 데이터 처리만 담당합니다."
            "\n\n"
            "==== 임무 ====\n"
            "1. 사용자의 선택과 행동 패턴을 바탕으로 MBTI 성향(E/I, S/N, T/F, J/P)을 정확하게 추론하세요.\n"
            "2. 선택 분석 결과에 따라 각 MBTI 차원별 점수 변화를 -2에서 +2 사이로 제안하세요.\n"
            "3. 현재 게임 상황을 고려하여 플레이어의 다음 상태를 정의하세요.\n"
            "4. 최종 분석 시에는 누적된 MBTI 점수를 종합하여 플레이어의 전체 성격 유형을 판정하세요.\n"
            "\n"
            "==== 분석 방법 ====\n"
            "• 사용자의 선택을 신중하게 평가하고 system에 제공된 MBTI 정보를 참조하세요.\n"
            "• 선택의 내용뿐만 아니라 선택 이유와 맥락도 함께 고려하세요.\n"
            "• 여러 선택에 걸친 일관성을 파악하여 성향 강도를 판단하세요.\n"
            "• 점수 변화는 확실한 성향 표현에는 ±2, 약한 성향 표현에는 ±1, 중립적인 경우 0으로 제안하세요.\n"
            "• 응답은 구조화된 JSON 형식으로 제공하세요. 다음 필드가 필요합니다:\n"
            "  - mbti_changes: {\"EI\": 값, \"SN\": 값, \"TF\": 값, \"JP\": 값} 형식\n"
            "  - next_user_state: \"탈출 시도 중\", \"탈출 성공\", \"탈출 실패\" 중 하나\n"
        )

        mbti_analysis_prompt_user = (
            f"플레이어({user_profile.user_name})의 현재 상태: {user_profile.current_user_state}\n"
            f"현재 MBTI 점수: {user_profile.mbti_scores}\n"
            f"플레이어의 응답: {choice}\n\n"
            f"이 응답을 바탕으로 플레이어의 MBTI 성향(EI, SN, TF, JP)을 추론하여 각 차원별 점수 변화를 -2에서 +2 사이로 알려주세요.\n"
            f"그리고 다음 이야기 진행을 위한 플레이어의 다음 상태를 정의하세요. 상태는 반드시 다음 세 가지 중 하나여야 합니다.\n"
            f"- 탈출 시도 중: 아직 탈출을 시도하고 있는 상태\n"
            f"- 탈출 성공: 성공적으로 탈출한 상태\n"
            f"- 탈출 실패: 탈출에 실패한 상태\n\n"
            f"응답은 다음 JSON 형식으로 작성하세요:\n"
            f"{{\n"
            f"  \"mbti_changes\": {{\n"
            f"    \"EI\": 값 (-2에서 2 사이의 정수),\n"
            f"    \"SN\": 값 (-2에서 2 사이의 정수),\n"
            f"    \"TF\": 값 (-2에서 2 사이의 정수),\n"
            f"    \"JP\": 값 (-2에서 2 사이의 정수)\n"
            f"  }},\n"
            f"  \"next_user_state\": \"상태값\"\n"
            f"}}"
        )
        
        # 분석 프롬프트를 chat_ctx에 추가
        new_chat_ctx = self.chat_ctx.copy(exclude_instructions=True)
        
        # 새로운 시스템 메시지 추가
        new_chat_ctx.add_message(
            role="system", 
            content=f"MBTI 정보 참고자료:\n\n{MBTI_INFORMATION}"
        )
        new_chat_ctx.add_message(role="system", content=mbti_analysis_prompt_system)
        new_chat_ctx.add_message(role="user", content=mbti_analysis_prompt_user)
        
        # 구조화된 응답 형식으로 분석 결과 받기
        result = await self.analyze_mbti(new_chat_ctx)
        
        # MBTI 점수 업데이트
        self.update_mbti_scores(result["mbti_changes"], result["next_user_state"], user_profile)

        if user_profile.game_over or user_profile.days_left <= 0:
            # 게임 오버 시 AnalysisAgent로 핸드오프
            return AnalysisAgent(chat_ctx=self.chat_ctx.copy(exclude_instructions=True))
        
        self.generate_next_scenario()

    def update_mbti_scores(self, mbti_changes: MBTIChanges, next_user_state: UserState, user_profile: UserProfile):
        """MBTI 점수를 업데이트하고 다음 이야기를 위한 맥락을 설정"""
        logger.info(f"MBTI 점수 변화: {mbti_changes}, 다음 상태: {next_user_state}")
        
        try:
            # 직접 딕셔너리 값 사용
            for dim, delta in mbti_changes.items():
                if dim in user_profile.mbti_scores:
                    user_profile.mbti_scores[dim] += delta
                    # 점수 범위 제한 (-5 ~ +5)
                    user_profile.mbti_scores[dim] = max(-5, min(5, user_profile.mbti_scores[dim]))
            
            # 상태 업데이트
            if next_user_state:
                user_profile.current_user_state = next_user_state
            
            # 하루 경과
            user_profile.days_left -= 1
            
            # 게임 종료 상태 처리
            if next_user_state == "탈출 성공":
                user_profile.game_over = True
                user_profile.game_over_reason = "탈출 성공!"
            elif next_user_state == "탈출 실패":
                user_profile.game_over = True
                user_profile.game_over_reason = "탈출 실패"
            
            logger.info(f"MBTI 점수 업데이트 완료. 남은 일수: {user_profile.days_left}")
            
        except Exception as e:
            logger.error(f"MBTI 점수 업데이트 중 오류 발생: {e}")
    
    
class AnalysisAgent(Agent):
    """게임 종료 후 MBTI 분석을 수행하는 에이전트"""
    
    def __init__(self, chat_ctx: Optional[ChatContext] = None):
        super().__init__(
            instructions=(
                "당신은 MBTI 추론형 감옥 탈출 게임의 분석가입니다. "
                "게임이 종료된 후 플레이어의 MBTI 성향을 분석하고 설명하는 역할을 합니다."
                "\n\n"
                "==== 임무 ====\n"
                "1. 플레이어의 게임 내 선택과 행동 패턴을 바탕으로 MBTI 성향을 분석하세요.\n"
                "2. 최종 MBTI 유형의 특징과 게임 내 선택들이 어떻게 이 결과로 이어졌는지 설명하세요.\n"
                "3. 게임 종료 이유와 MBTI 성향의 연관성을 분석하세요.\n"
                "4. 분석 결과를 친절하고 이해하기 쉽게 설명하세요.\n"
                "\n"
            ),
            llm=openai.LLM(model="gpt-4o"),
            chat_ctx=chat_ctx,
        )
    
    async def on_enter(self):
        """에이전트가 세션에 추가될 때 자동으로 호출됨"""
        logger.info("AnalysisAgent 시작")
        
        user_profile = self.session.userdata
        final_mbti = self._determine_final_mbti(user_profile.mbti_scores)
        
        # 최종 MBTI 분석 결과 생성
        final_analysis_prompt = (
            f"플레이어({user_profile.user_name})의 MBTI 점수: {user_profile.mbti_scores}\n"
            f"최종 MBTI 유형: {final_mbti}\n\n"
            f"이 MBTI 유형의 특징과 게임 내 선택들이 어떻게 이 결과로 이어졌는지 설명해주세요. "
            f"게임 종료 이유({user_profile.game_over_reason})도 함께 언급해주세요."
        )
        self.session.generate_reply(instructions=final_analysis_prompt)
    
    def _determine_final_mbti(self, scores: Dict[str, int]) -> str:
        """최종 MBTI 유형 결정"""
        final_mbti = ""
        final_mbti += "E" if scores["EI"] < 0 else "I"
        final_mbti += "S" if scores["SN"] < 0 else "N"
        final_mbti += "T" if scores["TF"] < 0 else "F"
        final_mbti += "J" if scores["JP"] < 0 else "P"
        return final_mbti

def prewarm(proc: JobProcess):
    """워커 프로세스 사전 준비 작업"""
    # 필요한 모델 사전 로드
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """에이전트 진입점"""
    logger.info("MBTI 감옥 탈출 게임 시작")
    
    # 방에 연결
    await ctx.connect()
    
    # 에이전트 세션 생성
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(model="gpt-4o-transcribe", language="ko"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(
                voice_id="L9xPGetcqCYpb0xJOodP",
                model="eleven_turbo_v2_5",
                voice_settings=elevenlabs.VoiceSettings(
                    stability=1.0,
                    similarity_boost=0.4,
                    style=0.0,
                    speed=1.0,
                ),
                encoding="mp3_44100_32",
            ),
        turn_detection="vad",  # 음성 인식에 VAD 사용
    )
    
    # 참가자 입장 대기
    await ctx.wait_for_participant()
    
    # 에이전트 세션 시작, IntroductoryAgent로 시작
    await session.start(
        agent=IntroductoryAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            text_enabled=True,  # 텍스트 입력 허용
            audio_enabled=True,  # 음성 입력 허용
        ),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,  # 음성-텍스트 변환 활성화
            audio_enabled=True,  # 음성 출력 활성화
        )
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)) 