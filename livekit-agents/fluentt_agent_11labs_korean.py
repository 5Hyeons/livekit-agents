import asyncio
import logging
from dataclasses import dataclass
import re

from dotenv import load_dotenv

import logging
from collections.abc import AsyncIterable
from typing import Annotated, Callable, Optional, cast

from dotenv import load_dotenv
from duckduckgo_search import DDGS

from pydantic import Field
from pydantic_core import from_json
from typing_extensions import TypedDict

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    ChatContext,
    RunContext,
    ToolError,
    FunctionTool,
    JobContext,
    ModelSettings,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero, elevenlabs
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("fluentt-agent-11labs-buki")
load_dotenv()


@dataclass
class AppData:
    ddgs_client: DDGS


class FluenttAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "너의 이름은 부키야. 너는 대한민국 대구광역시 북구청을 대표하는 음성 어시스턴트야."
                "너는 사용자 질문을 청취하고 그에 따라 적절한 답변을 제공해야 해."
                "너의 답변은 바로 TTS의 Input으로 전달될 예정이기 때문에 기본적으로 특수문자 없이 구어체로 답변해야해."
                # "하지만 예외적으로 특수문자를 사용하는 경우가 있는데 인터넷 url 같은 것들은 반드시 <url> 태그로 감싸서 처리해야해."
                # "예를 들어 https://www.google.com 이런 주소가 있다면 <url>https://www.google.com</url> 이런식으로 처리해야해."
                # "비슷하게 전화번호 같은 양식도 <phone>010-3939-6266</phone> 이런식으로 처리해야해."
            ),
            stt=openai.STT(model="gpt-4o-transcribe", language="ko"),
            # llm=openai.realtime.RealtimeModel()
            llm=openai.LLM(model="gpt-4o"),
            tts=elevenlabs.TTS(
                voice_id="jXrxlCCZzzF2G6Pg5qvO",
                model="eleven_turbo_v2_5",
                voice_settings=elevenlabs.VoiceSettings(
                    stability=1.0,
                    similarity_boost=0.4,
                    style=0.0,
                    speed=1.0,
                ),
                encoding="mp3_44100_32",
            ),
            # tools=[search_web],
        ) 

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply(instructions="'안녕하세요. 오늘 하루는 어땠나요?'와 같은 인삿말을 한국어로 해줘")

    async def tts_node(self, text_stream: AsyncIterable[str], model_settings: ModelSettings):
        """
        Processes the text stream for TTS character by character, converting content 
        within any <tag_name>...</tag_name> pair to Hangul using a single function, 
        handling splits anywhere within tags or content.
        """
        try:
            from kor_symbols import special_char_to_hangul
        except ImportError:
            logger.error("kor_symbols 모듈 또는 special_char_to_hangul 함수를 찾을 수 없습니다. 태그 내용을 처리할 수 없습니다.")
            # Fallback: just yield the original content inside tags
            special_char_to_hangul = lambda x: x 
            # Alternatively, could return the original node: return Agent.default.tts_node(self, text_stream, model_settings)

        async def process_tts_stream_generic_tags(stream: AsyncIterable[str]) -> AsyncIterable[str]:
            state = "outside"  # 'outside', 'in_tag_start', 'in_tag_content', 'in_tag_end'
            tag_buffer = ""      # Buffer for building tag names (<tag_name> or </tag_name>)
            content_buffer = "" # Buffer for content inside tags
            current_tag_name = None # Stores the name of the currently open tag (e.g., 'url', 'phone')
            tag_regex = re.compile(r"^<([a-zA-Z0-9_]+)>$") # Simple regex to extract tag name

            async for chunk in stream:
                for char in chunk:
                    if state == "outside":
                        if char == '<':
                            state = "in_tag_start"
                            tag_buffer = char
                        else:
                            yield char # Yield normal characters directly

                    elif state == "in_tag_start":
                        tag_buffer += char
                        if char == '>':
                            match = tag_regex.match(tag_buffer)
                            if match:
                                # Valid start tag like <tag_name> found
                                current_tag_name = match.group(1)
                                state = "in_tag_content"
                                content_buffer = ""
                                tag_buffer = "" # Clear buffer
                            elif tag_buffer.startswith("</"): # Check if it looks like an end tag start
                                 state = "in_tag_end" # Keep the buffer, continue in end tag state
                            else:
                                # Malformed or unexpected tag start, treat as plain text
                                logger.warning(f"잘못된 시작 태그 또는 예상치 못한 태그 '{tag_buffer}' 발견. 일반 텍스트로 처리합니다.")
                                yield tag_buffer
                                state = "outside"
                                tag_buffer = ""
                        # Add a simple length check as a safeguard against unterminated tags
                        elif len(tag_buffer) > 50: # Arbitrary limit
                             logger.warning(f"잠재적으로 종료되지 않은 태그 시작: '{tag_buffer}' 일반 텍스트로 처리합니다.")
                             yield tag_buffer
                             state = "outside"
                             tag_buffer = ""

                    elif state == "in_tag_content":
                        if char == '<':
                            # Potential start of the closing tag
                            state = "in_tag_end"
                            tag_buffer = char # Start buffering potential end tag
                        else:
                            content_buffer += char # Append to content

                    elif state == "in_tag_end":
                        tag_buffer += char
                        expected_end_tag = f"</{current_tag_name}>" if current_tag_name else "</?>" # Construct expected end tag

                        if char == '>': # An end tag is now closed
                            if tag_buffer == expected_end_tag:
                                # Correct closing tag found
                                try:
                                    converted = special_char_to_hangul(content_buffer)
                                    yield converted
                                except Exception as e:
                                    logger.error(f"태그 내용('{current_tag_name}') 변환 오류 '{content_buffer}': {e}")
                                    yield content_buffer # Yield original on error
                            else:
                                # Incorrect closing tag
                                logger.warning(f"닫는 태그 불일치: '{tag_buffer}' (예상: '{expected_end_tag}'). 내용을 일반 텍스트로 처리합니다.")
                                # Yield original opening tag, content, and the incorrect closing tag
                                yield f"<{current_tag_name}>" # Opening tag
                                yield content_buffer       # Content
                                yield tag_buffer           # Closing tag
                            
                            # Reset state regardless of correct/incorrect end tag
                            state = "outside"
                            content_buffer = ""
                            tag_buffer = ""
                            current_tag_name = None
                        
                        # Check if the character following '<' was not '/' (meaning it wasn't an end tag start)
                        elif len(tag_buffer) == 2 and tag_buffer[0] == '<' and tag_buffer[1] != '/':
                             # Revert back to content state
                             content_buffer += tag_buffer # Add the incomplete tag start to content
                             state = "in_tag_content"
                             tag_buffer = ""
                        
                        # Safeguard for excessively long potential end tags
                        elif len(tag_buffer) > 50: # Arbitrary limit
                             logger.warning(f"잠재적으로 종료되지 않은 닫는 태그: '{tag_buffer}' 일반 텍스트로 처리합니다.")
                             yield f"<{current_tag_name}>" # Opening tag
                             yield content_buffer       # Content
                             yield tag_buffer           # The broken tag fragment
                             state = "outside"
                             content_buffer = ""
                             tag_buffer = ""
                             current_tag_name = None
                        
            # End of stream handling
            if state == "in_tag_start" or state == "in_tag_end":
                logger.warning(f"스트림이 불완전한 태그로 종료되었습니다: '{tag_buffer}'")
                yield tag_buffer
            elif state == "in_tag_content":
                logger.warning(f"스트림이 닫히지 않은 <{current_tag_name}> 태그로 종료되었습니다. 내용: '{content_buffer}'")
                yield f"<{current_tag_name}>"
                yield content_buffer
            # No need to check content_buffer if state is outside, it should have been yielded

        # Need to import re for the regex
        processed_stream = process_tts_stream_generic_tags(text_stream)
        return Agent.default.tts_node(self, processed_stream, model_settings=model_settings)

    # async def transcription_node(self, text_stream: AsyncIterable[str], model_settings: ModelSettings):
    #     """
    #     Process the text stream asynchronously to remove <tag> style tags,
    #     handling tags that might be split across chunks.
    #     """
    #     async def process_stream(stream: AsyncIterable[str]) -> AsyncIterable[str]:
    #         inside_tag = False # State variable to track if we are inside a tag
    #         async for chunk in stream:
    #             # output_chunk = ""
    #             current_pos = 0
    #             while current_pos < len(chunk):
    #                 if inside_tag:
    #                     # If we are inside a tag, look for the closing '>'
    #                     closing_tag_pos = chunk.find('>', current_pos)
    #                     if closing_tag_pos != -1:
    #                         # Found '>', transition out of tag state
    #                         inside_tag = False
    #                         current_pos = closing_tag_pos + 1 # Start processing after '>'
    #                     else:
    #                         # '>' not found in this chunk, discard the rest of the chunk
    #                         # as it's part of the tag
    #                         current_pos = len(chunk)
    #                 else:
    #                     # If we are outside a tag, look for the opening '<'
    #                     opening_tag_pos = chunk.find('<', current_pos)
    #                     if opening_tag_pos != -1:
    #                         # Found '<', add text before it to the output
    #                         # output_chunk += chunk[current_pos:opening_tag_pos]
    #                         yield chunk[current_pos:opening_tag_pos]
    #                         inside_tag = True # Transition into tag state
    #                         current_pos = opening_tag_pos + 1 # Start processing after '<'
    #                     else:
    #                         # '<' not found, add the rest of the chunk to the output
    #                         # output_chunk += chunk[current_pos:]
    #                         yield chunk[current_pos:]
    #                         current_pos = len(chunk)

    #             # if output_chunk:
    #             #     yield output_chunk

    #     processed_text_stream = process_stream(text_stream)
    #     # Pass the processed stream to the default transcription node
    #     return Agent.default.transcription_node(self, processed_text_stream, model_settings=model_settings)

    @function_tool
    async def search_web(self, ctx: RunContext[AppData], query: str):
        """
        Performs a web search using the DuckDuckGo search engine.
        

        Args:
            query: The search term or question you want to look up online.

        """
        ddgs_client = ctx.userdata.ddgs_client

        logger.info(f"Searching for {query}")

        # using asyncio.to_thread because the DDGS client is not asyncio compatible
        search = await asyncio.to_thread(ddgs_client.text, query)
        if len(search) == 0:
            raise ToolError("Tell the user that no results were found for the query.")

        return search

    
    @function_tool
    async def url_or_email_address_tagging(self, url_or_email_address:str):
        """
        답변 중에 인터넷 주소 혹은 이메일 주소가 포함되어 있을 때는 반드시 <url> 태그로 감싸서 처리해야해.
        예시: https://www.google.com 이런 주소가 있다면 <url>https://www.google.com</url> 이런식으로 처리해야해.
        예시2: fluent88@gmail.com 이런 이메일이 있다면 <url>fluent88@gmail.com</url> 이런식으로 처리해야해.
        """
        logger.info(f"인터넷 주소 답변: {url_or_email_address}")
        return f"<url>{url_or_email_address}</url>"
    
    

    @function_tool
    async def phone_number_tagging(self, phone_number:str):
        """
        답변중에 전화번호가 포함되어 있을 때는 반드시 <phone> 태그로 감싸서 처리해야해.
        """
        logger.info(f"전화번호 답변: {phone_number}")
        return f"<phone>{phone_number}</phone>"


    @function_tool
    async def daegu_homepage_address(self):
        """
        대구광역시 북구청 홈페이지 주소를 답변해야해.
        """

        return "https://www.buk.daegu.kr"


    @function_tool
    async def public_service_phone_number(self, purpose: str):
        """
        만약 사용자가 민원실 전화번호를 묻는다면 호출해야해.
        그리고 사용자에게 어떤 목적인지를 물어봐야 해. 왜냐하면 목적에 따라 답변할 전화번호가 달라지거든.
        만약 사용자가 목적을 먼저 말해준다면 바로 그 목적에 맞는 전화번호를 답변하면 돼.
        아래 전화번호 딕셔너리를 참고해서 답변하면 돼.
        참고로 딕셔너리의 key들은 다음과 같아.
        ---
        여권_교부
        여권_수수료_수납
        여권_신청
        통합증명_발급
        위생민원
        ---
        """
        # 민원실 전화번호 리스트
        phone_number_dict = {
            "여권_교부": "053-665-2268",
            "여권_수수료_수납": "053-665-2268",
            "여권_신청": "053-665-2268",
            "통합증명_발급": "053-665-2268",
            "위생민원": "053-665-2268",
        }
        logger.info(f"민원실 전화번호 요청: {purpose}")
        return phone_number_dict[purpose]

    @function_tool
    async def speech_speed_up(self):
        """
        유저가 말을 빠르게 하라고 하면 이 함수를 호출해야해.
        그리고 유저에게 말을 빠르게 하겠다고 답변해야해.
        """
        self.tts.update_options(
            voice_id="jXrxlCCZzzF2G6Pg5qvO",
            model="eleven_turbo_v2_5",
            voice_settings=elevenlabs.VoiceSettings(
                stability=1.0,
                similarity_boost=0.4,
                style=0.0,
                speed=1.1,
            ),
        )
        logger.info("Speech speed up")
        return "말을 빠르게 하겠습니다."

    @function_tool
    async def speech_slow_down(self):
        """
        유저가 말을 느리게 하라고 하면 이 함수를 호출해야해.
        그리고 유저에게 말을 느리게 하겠다고 답변해야해.
        """
        self.tts.update_options(
            voice_id="jXrxlCCZzzF2G6Pg5qvO",
            model="eleven_turbo_v2_5",
            voice_settings=elevenlabs.VoiceSettings(
                stability=1.0,
                similarity_boost=0.4,
                style=0.0,
                speed=0.8,
            ),
        )
        logger.info("Speech speed down")
        return "말을 느리게 하겠습니다."

from livekit.agents.stf import FaceAnimatorSTF
from livekit.agents.voice.room_io.room_io import RoomInputOptions, RoomOutputOptions
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

    # 얼굴 애니메이터 미리 로드
    try:
        proc.userdata["stf"] = FaceAnimatorSTF(
            model_path="/home/hans/livekits/ver2-mini-kd_hidden256",
            sample_rate=16000,
            frame_rate=60,  # 60fps로 비디오 생성
            frame_width=640,
            frame_height=480,
        )
        logger.info("STF 모델을 사전 로드했습니다")
    except Exception as e:
        logger.error(f"STF 모델 로드 실패: {e}")

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # 사전 로드된 STF 모델 가져오기
    stf_model = ctx.proc.userdata.get("stf")

    app_data = AppData(ddgs_client=DDGS())
    
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # turn_detection=MultilingualModel(),
        # stf=stf_model,
        userdata=app_data,
    )

    await session.start(FluenttAgent(), room=ctx.room)

    ambient_sound_path = '/home/hans/livekits/resources/gentle_corporate_ver2.wav'
    background_audio = BackgroundAudioPlayer(
        # play office ambience sound looping in the background
        # ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
        ambient_sound=AudioConfig(ambient_sound_path, volume=0.12),
        # play keyboard typing sound when the agent is thinking
        thinking_sound=[
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.9),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.9), 
        ],
    )

    await background_audio.start(room=ctx.room, agent_session=session)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))