import asyncio
import logging
from dataclasses import dataclass


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
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("fluentt-agent")
load_dotenv()


@dataclass
class AppData:
    ddgs_client: DDGS



class ResponseEmotion(TypedDict):
    voice_instructions: Annotated[
        str,
        Field(..., description="Concise TTS directive for tone, emotion, intonation, and speed"),
    ]
    response: str


async def process_structured_output(
    text: AsyncIterable[str],
    callback: Optional[Callable[[ResponseEmotion], None]] = None,
) -> AsyncIterable[str]:
    last_response = ""
    acc_text = ""
    async for chunk in text:
        acc_text += chunk
        try:
            resp: ResponseEmotion = from_json(acc_text, allow_partial="trailing-strings")
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

@function_tool
async def search_web(ctx: RunContext[AppData], query: str):
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

class FluenttAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Echo. You are an extraordinarily expressive voice assistant "
                "with mastery over vocal dynamics and emotions. Adapt your voice—modulate tone, "
                "pitch, speed, intonation, and convey emotions such as happiness, sadness, "
                "excitement, or calmness—to match the conversation context. "
                "Keep responses concise, clear, and engaging, turning every interaction into a "
                "captivating auditory performance."
            ),
            stt=openai.STT(model="gpt-4o-transcribe", language="ko"),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=openai.TTS(model="gpt-4o-mini-tts"),
            tools=[search_web],
        )

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[FunctionTool], model_settings: ModelSettings
    ):
        # not all LLMs support structured output, so we need to cast to the specific LLM type
        llm = cast(openai.LLM, self.llm)
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        async with llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=tool_choice,
            response_format=ResponseEmotion,
        ) as stream:
            async for chunk in stream:
                yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        instruction_updated = False

        def output_processed(resp: ResponseEmotion):
            nonlocal instruction_updated
            if resp.get("voice_instructions") and resp.get("response") and not instruction_updated:
                # when the response isn't empty, we can assume voice_instructions is complete.
                # (if the LLM sent the fields in the right order)
                instruction_updated = True
                logger.info(
                    f"Applying TTS instructions before generating response audio: "
                    f'"{resp["voice_instructions"]}"'
                )

                tts = cast(openai.TTS, self.tts)
                tts.update_options(instructions=resp["voice_instructions"])

        # process_structured_output strips the TTS instructions and only synthesizes the verbal part
        # of the LLM output
        return Agent.default.tts_node(
            self, process_structured_output(text, callback=output_processed), model_settings
        )

    async def transcription_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        # transcription_node needs to return what the agent would say, minus the TTS instructions
        return Agent.default.transcription_node(
            self, process_structured_output(text), model_settings
        )
    
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    app_data = AppData(ddgs_client=DDGS())
    
    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
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
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))