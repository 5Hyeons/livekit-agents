from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import time
import functools

from pydantic import ValidationError

from livekit import rtc

from .. import debug, llm, utils
from ..llm import (
    ChatChunk,
    ChatContext,
    StopResponse,
    ToolContext,
    ToolError,
    utils as llm_utils,
)
from ..llm.tool_context import (
    is_function_tool,
    is_raw_function_tool,
)
from ..log import logger
from ..types import NotGivenOr
from ..utils import aio
from . import io
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent import Agent, ModelSettings
    from .agent_session import AgentSession


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self): ...


@dataclass
class _LLMGenerationData:
    text_ch: aio.Chan[str]
    function_ch: aio.Chan[llm.FunctionCall]
    generated_text: str = ""
    generated_functions: list[llm.FunctionCall] = field(default_factory=list)
    id: str = field(default_factory=lambda: utils.shortuuid("item_"))


def perform_llm_inference(
    *,
    node: io.LLMNode,
    chat_ctx: ChatContext,
    tool_ctx: ToolContext,
    model_settings: "ModelSettings",
) -> tuple[asyncio.Task, _LLMGenerationData]:
    text_ch = aio.Chan()
    function_ch = aio.Chan()

    data = _LLMGenerationData(text_ch=text_ch, function_ch=function_ch)

    @utils.log_exceptions(logger=logger)
    async def _inference_task():
        tools = list(tool_ctx.function_tools.values())
        llm_node = node(
            chat_ctx,
            tools,
            model_settings,
        )
        if asyncio.iscoroutine(llm_node):
            llm_node = await llm_node

        # update the tool context after llm node
        tool_ctx.update_tools(tools)

        if isinstance(llm_node, str):
            data.generated_text = llm_node
            text_ch.send_nowait(llm_node)
            return True

        if isinstance(llm_node, AsyncIterable):
            # forward llm stream to output channels
            try:
                async for chunk in llm_node:
                    # io.LLMNode can either return a string or a ChatChunk
                    if isinstance(chunk, str):
                        data.generated_text += chunk
                        text_ch.send_nowait(chunk)

                    elif isinstance(chunk, ChatChunk):
                        if not chunk.delta:
                            continue

                        if chunk.delta.tool_calls:
                            for tool in chunk.delta.tool_calls:
                                if tool.type != "function":
                                    continue

                                fnc_call = llm.FunctionCall(
                                    id=f"{data.id}/fnc_{len(data.generated_functions)}",
                                    call_id=tool.call_id,
                                    name=tool.name,
                                    arguments=tool.arguments,
                                )
                                data.generated_functions.append(fnc_call)
                                function_ch.send_nowait(fnc_call)

                        if chunk.delta.content:
                            data.generated_text += chunk.delta.content
                            text_ch.send_nowait(chunk.delta.content)
                    else:
                        logger.warning(
                            f"LLM node returned an unexpected type: {type(chunk)}",
                        )
            finally:
                if isinstance(llm_node, _ACloseable):
                    await llm_node.aclose()

            return True

        return False

    llm_task = asyncio.create_task(_inference_task())
    llm_task.add_done_callback(lambda _: text_ch.close())
    llm_task.add_done_callback(lambda _: function_ch.close())
    return llm_task, data


@dataclass
class _TTSGenerationData:
    audio_ch: aio.Chan[rtc.AudioFrame]


def perform_tts_inference(
    *, node: io.TTSNode, input: AsyncIterable[str], model_settings: "ModelSettings"
) -> tuple[asyncio.Task, _TTSGenerationData]:
    audio_ch = aio.Chan[rtc.AudioFrame]()

    @utils.log_exceptions(logger=logger)
    async def _inference_task():
        tts_node = node(input, model_settings)
        if asyncio.iscoroutine(tts_node):
            tts_node = await tts_node

        if isinstance(tts_node, AsyncIterable):
            async for audio_frame in tts_node:
                audio_ch.send_nowait(audio_frame)

            return True

        return False

    tts_task = asyncio.create_task(_inference_task())
    tts_task.add_done_callback(lambda _: audio_ch.close())

    return tts_task, _TTSGenerationData(audio_ch=audio_ch)


@dataclass
class _STFGenerationData:
    """STF 생성 데이터."""
    anim_ch: aio.Chan[io.AnimationData]


def perform_stf_inference(
    *,
    node: Any,
    input: AsyncIterable[rtc.AudioFrame],
    model_settings: "ModelSettings",
) -> tuple[asyncio.Task, _STFGenerationData]:
    """
    STF 모델 추론을 수행하고 애니메이션 데이터를 생성합니다.
    
    Args:
        node: STF 모델 노드 (Agent.stf_node)
        input: 오디오 프레임 스트림
        model_settings: 모델 설정
        
    Returns:
        tuple[asyncio.Task, _STFGenerationData]: 
            STF 생성 작업과 STF 생성 데이터를 포함하는 튜플
    """
    anim_ch = aio.Chan[io.AnimationData]()
    out = _STFGenerationData(anim_ch=anim_ch)
    task = asyncio.create_task(_stf_inference_task(node, input, anim_ch, model_settings))
    return task, out


@utils.log_exceptions(logger=logger)
async def _stf_inference_task(
    node: Any,
    input: AsyncIterable[rtc.AudioFrame],
    anim_ch: aio.Chan[io.AnimationData],
    model_settings: "ModelSettings",
) -> None:
    """
    STF 모델 추론을 수행하고 결과를 채널에 전달하는 작업입니다.
    
    Args:
        node: STF 모델 노드 (Agent.stf_node)
        input: 오디오 프레임 스트림
        anim_ch: 애니메이션 데이터 출력 채널
        model_settings: 모델 설정
    """
    frames_count = 0
    start_time = time.time()
    logger.info("STF 추론 작업 시작")
    
    try:
        # node가 코루틴인 경우 실행
        logger.debug("STF 노드 실행 중")
        stf_result = node(input, model_settings)
        if asyncio.iscoroutine(stf_result):
            stf_result = await stf_result
            
        # 애니메이션 데이터 생성이 없는 경우
        if stf_result is None:
            logger.warning("STF 결과가 없습니다. 애니메이션 데이터가 생성되지 않았습니다.")
            return
            
        # 각 애니메이션 데이터를 채널로 전달
        logger.debug("애니메이션 데이터 스트림 처리 시작")
        async for anim_data in stf_result:
            anim_ch.send_nowait(anim_data)
            frames_count += 1
            # if frames_count % 30 == 0:  # 30 프레임마다 로그 출력
            #     logger.debug(f"STF 애니메이션 데이터 전송 중: {frames_count}개 프레임 처리됨")
    except Exception as e:
        logger.error(f"STF 추론 중 오류 발생: {e}", exc_info=True)
        raise
    finally:
        duration = time.time() - start_time
        # logger.info(f"STF 추론 작업 완료: {frames_count}개 애니메이션 프레임 생성, 소요 시간: {duration:.2f}초")
        anim_ch.close()


@dataclass
class _TextOutput:
    text: str
    first_text_fut: asyncio.Future


def perform_text_forwarding(
    *, text_output: io.TextOutput | None, source: AsyncIterable[str]
) -> tuple[asyncio.Task, _TextOutput]:
    out = _TextOutput(text="", first_text_fut=asyncio.Future())
    task = asyncio.create_task(_text_forwarding_task(text_output, source, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _text_forwarding_task(
    text_output: io.TextOutput | None,
    source: AsyncIterable[str],
    out: _TextOutput,
) -> None:
    try:
        async for delta in source:
            out.text += delta
            if text_output is not None:
                await text_output.capture_text(delta)

            if not out.first_text_fut.done():
                out.first_text_fut.set_result(None)
    finally:
        if isinstance(source, _ACloseable):
            await source.aclose()

        if text_output is not None:
            text_output.flush()


@dataclass
class _AudioOutput:
    audio: list[rtc.AudioFrame]
    first_frame_fut: asyncio.Future


def perform_audio_forwarding(
    *,
    audio_output: io.AudioOutput,
    tts_output: AsyncIterable[rtc.AudioFrame],
) -> tuple[asyncio.Task, _AudioOutput]:
    out = _AudioOutput(audio=[], first_frame_fut=asyncio.Future())
    task = asyncio.create_task(_audio_forwarding_task(audio_output, tts_output, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _audio_forwarding_task(
    audio_output: io.AudioOutput,
    tts_output: AsyncIterable[rtc.AudioFrame],
    out: _AudioOutput,
) -> None:
    resampler: rtc.AudioResampler | None = None
    try:
        async for frame in tts_output:
            out.audio.append(frame)

            if (
                not out.first_frame_fut.done()
                and audio_output.sample_rate is not None
                and frame.sample_rate != audio_output.sample_rate
                and resampler is None
            ):
                resampler = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=audio_output.sample_rate,
                    num_channels=frame.num_channels,
                )

            if resampler:
                for f in resampler.push(frame):
                    await audio_output.capture_frame(f)
            else:
                await audio_output.capture_frame(frame)

            # set the first frame future if not already set
            # (after completing the first frame)
            if not out.first_frame_fut.done():
                out.first_frame_fut.set_result(None)
    finally:
        if isinstance(tts_output, _ACloseable):
            await tts_output.aclose()

        if resampler:
            for frame in resampler.flush():
                await audio_output.capture_frame(frame)

        audio_output.flush()


@dataclass
class _ToolOutput:
    output: list[_PythonOutput]
    first_tool_fut: asyncio.Future


def perform_tool_executions(
    *,
    session: AgentSession,
    speech_handle: SpeechHandle,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[llm.ToolChoice],
    function_stream: AsyncIterable[llm.FunctionCall],
) -> tuple[asyncio.Task, _ToolOutput]:
    tool_output = _ToolOutput(output=[], first_tool_fut=asyncio.Future())
    task = asyncio.create_task(
        _execute_tools_task(
            session=session,
            speech_handle=speech_handle,
            tool_ctx=tool_ctx,
            tool_choice=tool_choice,
            function_stream=function_stream,
            tool_output=tool_output,
        ),
        name="execute_tools_task",
    )
    return task, tool_output


@utils.log_exceptions(logger=logger)
async def _execute_tools_task(
    *,
    session: AgentSession,
    speech_handle: SpeechHandle,
    tool_ctx: ToolContext,
    tool_choice: NotGivenOr[llm.ToolChoice],
    function_stream: AsyncIterable[llm.FunctionCall],
    tool_output: _ToolOutput,
) -> None:
    """execute tools, when cancelled, stop executing new tools but wait for the pending ones"""

    from .agent import _authorize_inline_task
    from .events import RunContext

    tasks: list[asyncio.Task] = []
    try:
        async for fnc_call in function_stream:
            if tool_choice == "none":
                logger.error(
                    "received a tool call with tool_choice set to 'none', ignoring",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            # TODO(theomonnom): assert other tool_choice values

            if (function_tool := tool_ctx.function_tools.get(fnc_call.name)) is None:
                logger.warning(
                    f"unknown AI function `{fnc_call.name}`",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            if not is_function_tool(function_tool) and not is_raw_function_tool(function_tool):
                logger.error(
                    f"unknown tool type: {type(function_tool)}",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            try:
                json_args = fnc_call.arguments or "{}"
                fnc_args, fnc_kwargs = llm_utils.prepare_function_arguments(
                    fnc=function_tool,
                    json_arguments=json_args,
                    call_ctx=RunContext(
                        session=session, speech_handle=speech_handle, function_call=fnc_call
                    ),
                )

            except (ValidationError, ValueError):
                logger.exception(
                    f"tried to call AI function `{fnc_call.name}` with invalid arguments",
                    extra={
                        "function": fnc_call.name,
                        "arguments": fnc_call.arguments,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            if not tool_output.first_tool_fut.done():
                tool_output.first_tool_fut.set_result(None)

            logger.debug(
                "executing tool",
                extra={
                    "function": fnc_call.name,
                    "arguments": fnc_call.arguments,
                    "speech_id": speech_handle.id,
                },
            )

            py_out = _PythonOutput(fnc_call=fnc_call, output=None, exception=None)
            try:
                task = asyncio.create_task(
                    function_tool(*fnc_args, **fnc_kwargs),
                    name=f"function_tool_{fnc_call.name}",
                )

                tasks.append(task)
                _authorize_inline_task(task, function_call=fnc_call)
            except Exception:
                # catching exceptions here because even though the function is asynchronous,
                # errors such as missing or incompatible arguments can still occur at
                # invocation time.
                logger.exception(
                    "exception occurred while executing tool",
                    extra={
                        "function": fnc_call.name,
                        "speech_id": speech_handle.id,
                    },
                )
                continue

            def _log_exceptions(
                task: asyncio.Task,
                py_out: _PythonOutput,
                fnc_call: llm.FunctionCall,
            ) -> None:
                if task.cancelled():
                    # 선택 사항: 취소 시 tool_output에 추가할지 여부 결정
                    # py_out.exception = asyncio.CancelledError()
                    # tool_output.output.append(py_out)
                    logger.debug(f"Tool task '{fnc_call.name}' was cancelled.")
                    return

                if exc := task.exception():
                    py_out.exception = exc
                    logger.exception(
                        f"exception from AI function `{fnc_call.name}`",
                        extra={
                            "function": fnc_call.name,
                            "arguments": fnc_call.arguments,
                            "speech_id": speech_handle.id,
                        },
                    )

                    # MCP 연결 오류 처리 로직 추가
                    if isinstance(exc, ToolError) and \
                       "MCP tool" in exc.message and \
                       "network-related issue" in exc.message:
                        
                        logger.warning(f"MCP Connection-like error detected for tool '{fnc_call.name}'. Attempting to handle via AgentSession.")
                        
                        asyncio.create_task(
                            session._handle_mcp_server_error(error=exc, tool_name=fnc_call.name)
                        )
                else:
                    # 예외가 없는 경우 (성공)
                    try:
                        py_out.output = task.result()
                        logger.debug(f"Tool '{fnc_call.name}' executed successfully, result: {py_out.output}")
                    except asyncio.InvalidStateError:
                        # 아직 result()를 호출할 수 없는 경우 (매우 드문 케이스, 일반적으로 done callback은 task 완료 후 호출됨)
                        logger.error(f"Task '{fnc_call.name}' for tool in done callback but InvalidStateError on result().")
                        py_out.exception = RuntimeError("Task finished but result not available") # 임시 예외 설정
                
                # 성공했든, (처리 가능한) 예외가 발생했든 tool_output에 추가
                tool_output.output.append(py_out)
                # tasks.remove(task) # 이 부분은 원래 콜백 바깥에 있었을 가능성이 높음

            task.add_done_callback(
                functools.partial(_log_exceptions, py_out=py_out, fnc_call=fnc_call)
            )

        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    except asyncio.CancelledError:
        if len(tasks) > 0:
            names = [task.get_name() for task in tasks]
            logger.debug(
                "waiting for function call to finish before fully cancelling",
                extra={
                    "functions": names,
                    "speech_id": speech_handle.id,
                },
            )
            debug.Tracing.log_event(
                "waiting for function call to finish before fully cancelling",
                {
                    "functions": names,
                    "speech_id": speech_handle.id,
                },
            )
            await asyncio.gather(*tasks)
    finally:
        await utils.aio.cancel_and_wait(*tasks)

        if len(tool_output.output) > 0:
            logger.debug(
                "tools execution completed",
                extra={"speech_id": speech_handle.id},
            )
            debug.Tracing.log_event(
                "tools execution completed",
                {"speech_id": speech_handle.id},
            )


def _is_valid_function_output(value: Any) -> bool:
    VALID_TYPES = (str, int, float, bool, complex, type(None))

    if isinstance(value, VALID_TYPES):
        return True
    elif (
        isinstance(value, list)
        or isinstance(value, set)
        or isinstance(value, frozenset)
        or isinstance(value, tuple)
    ):
        return all(_is_valid_function_output(item) for item in value)
    elif isinstance(value, dict):
        return all(
            isinstance(key, VALID_TYPES) and _is_valid_function_output(val)
            for key, val in value.items()
        )
    return False


@dataclass
class _SanitizedOutput:
    fnc_call: llm.FunctionCall
    fnc_call_out: llm.FunctionCallOutput | None
    agent_task: Agent | None
    reply_required: bool = field(default=True)


@dataclass
class _PythonOutput:
    fnc_call: llm.FunctionCall
    output: Any
    exception: BaseException | None

    def sanitize(self) -> _SanitizedOutput:
        from .agent import Agent
        logger.debug(f"Sanitizing _PythonOutput for tool '{self.fnc_call.name}'. Output: {self.output}, Exception: {self.exception}")

        if isinstance(self.exception, ToolError):
            sanitized_output = _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=llm.FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output=self.exception.message,
                    is_error=True,
                ),
                agent_task=None,
            )
            logger.debug(f"Sanitized output (ToolError): {sanitized_output.fnc_call_out}")
            return sanitized_output

        if isinstance(self.exception, StopResponse):
            sanitized_output = _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=None,
                agent_task=None,
            )
            logger.debug(f"Sanitized output (StopResponse): {sanitized_output.fnc_call_out}")
            return sanitized_output

        if self.exception is not None:
            sanitized_output = _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=llm.FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output="An internal error occurred",  # Don't send the actual error message, as it may contain sensitive information  # noqa: E501
                    is_error=True,
                ),
                agent_task=None,
            )
            logger.debug(f"Sanitized output (Other Exception): {sanitized_output.fnc_call_out}")
            return sanitized_output

        task: Agent | None = None
        fnc_out: Any = self.output
        if (
            isinstance(self.output, list)
            or isinstance(self.output, set)
            or isinstance(self.output, frozenset)
            or isinstance(self.output, tuple)
        ):
            agent_tasks = [item for item in self.output if isinstance(item, Agent)]
            other_outputs = [item for item in self.output if not isinstance(item, Agent)]
            if len(agent_tasks) > 1:
                logger.error(
                    f"AI function `{self.fnc_call.name}` returned multiple AgentTask instances, ignoring the output",  # noqa: E501
                    extra={
                        "call_id": self.fnc_call.call_id,
                        "output": self.output,
                    },
                )

                return _SanitizedOutput(
                    fnc_call=self.fnc_call.model_copy(),
                    fnc_call_out=None,
                    agent_task=None,
                )

            task = next(iter(agent_tasks), None)

            # fmt: off
            fnc_out = (
                other_outputs if task is None
                else None if not other_outputs
                else other_outputs[0] if len(other_outputs) == 1
                else other_outputs
            )
            # fmt: on

        elif isinstance(fnc_out, Agent):
            task = fnc_out
            fnc_out = None

        if not _is_valid_function_output(fnc_out):
            logger.error(
                f"AI function `{self.fnc_call.name}` returned an invalid output",
                extra={
                    "call_id": self.fnc_call.call_id,
                    "output": self.output,
                },
            )
            return _SanitizedOutput(
                fnc_call=self.fnc_call.model_copy(),
                fnc_call_out=None,
                agent_task=None,
            )

        sanitized_output = _SanitizedOutput(
            fnc_call=self.fnc_call.model_copy(),
            fnc_call_out=(
                llm.FunctionCallOutput(
                    name=self.fnc_call.name,
                    call_id=self.fnc_call.call_id,
                    output=str(fnc_out or ""),  # take the string representation of the output
                    is_error=False,
                )
            ),
            reply_required=fnc_out is not None,  # require a reply if the tool returned an output
            agent_task=task,
        )
        logger.debug(f"Sanitized output (Success): {sanitized_output.fnc_call_out}")
        return sanitized_output


INSTRUCTIONS_MESSAGE_ID = "lk.agent_task.instructions"  #  value must not change
"""
The ID of the instructions message in the chat context. (only for stateless LLMs)
"""


def update_instructions(chat_ctx: ChatContext, *, instructions: str, add_if_missing: bool) -> None:
    """
    Update the instruction message in the chat context or insert a new one if missing.

    This function looks for an existing instruction message in the chat context using the identifier
    'INSTRUCTIONS_MESSAGE_ID'.

    Raises:
        ValueError: If an existing instruction message is not of type "message".
    """
    idx = chat_ctx.index_by_id(INSTRUCTIONS_MESSAGE_ID)
    if idx is not None:
        if chat_ctx.items[idx].type == "message":
            # create a new instance to avoid mutating the original
            chat_ctx.items[idx] = llm.ChatMessage(
                id=INSTRUCTIONS_MESSAGE_ID, role="system", content=[instructions]
            )
        else:
            raise ValueError(
                "expected the instructions inside the chat_ctx to be of type 'message'"
            )
    elif add_if_missing:
        # insert the instructions at the beginning of the chat context
        chat_ctx.items.insert(
            0, llm.ChatMessage(id=INSTRUCTIONS_MESSAGE_ID, role="system", content=[instructions])
        )


def remove_instructions(chat_ctx: ChatContext) -> None:
    # loop in case there are items with the same id (shouldn't happen!)
    while True:
        if msg := chat_ctx.get_by_id(INSTRUCTIONS_MESSAGE_ID):
            chat_ctx.items.remove(msg)
        else:
            break


STANDARD_SPEECH_RATE = 0.5  # words per second


def truncate_message(*, message: str, played_duration: float) -> str:
    # TODO(theomonnom): this is very naive
    from ..tokenize import _basic_word

    words = _basic_word.split_words(message, ignore_punctuation=False)
    total_duration = len(words) * STANDARD_SPEECH_RATE

    if total_duration <= played_duration:
        return message

    max_words = int(played_duration // STANDARD_SPEECH_RATE)
    if max_words < 1:
        return ""

    _, _, end_pos = words[max_words - 1]
    return message[:end_pos]


@dataclass
class _AnimationOutput:
    """애니메이션 데이터 출력 클래스"""
    animation: list[io.AnimationData]
    first_frame_fut: asyncio.Future


def perform_animation_forwarding(
    *,
    animation_output: io.AnimationDataOutput,
    stf_output: AsyncIterable[io.AnimationData],
) -> tuple[asyncio.Task, _AnimationOutput]:
    """
    STF에서 생성된 애니메이션 데이터를 출력으로 전달합니다.
    
    Args:
        animation_output: 애니메이션 데이터 출력 싱크
        stf_output: STF 추론에서 생성된 애니메이션 데이터 스트림
    
    Returns:
        asyncio.Task: 애니메이션 데이터 전달 태스크
        _AnimationOutput: 애니메이션 출력 데이터
    """
    out = _AnimationOutput(animation=[], first_frame_fut=asyncio.Future())
    task = asyncio.create_task(_animation_forwarding_task(animation_output, stf_output, out))
    return task, out


@utils.log_exceptions(logger=logger)
async def _animation_forwarding_task(
    animation_output: io.AnimationDataOutput,
    stf_output: AsyncIterable[io.AnimationData],
    out: _AnimationOutput,
) -> None:
    """
    STF에서 생성된 애니메이션 데이터를 애니메이션 출력으로 전달하는
    작업을 수행합니다.
    
    Args:
        animation_output: 애니메이션 데이터 출력 싱크
        stf_output: STF 추론에서 생성된 애니메이션 데이터 스트림
        out: 애니메이션 출력 데이터
    """
    frames_count = 0
    start_time = time.time()
    logger.info("애니메이션 데이터 전달 작업 시작")
    
    try:
        async for anim_data in stf_output:
            out.animation.append(anim_data)
            await animation_output.capture_frame(anim_data)
            frames_count += 1
            
            if frames_count == 1:
                logger.info("첫 번째 애니메이션 프레임 전송 완료")
            
            # if frames_count % 120 == 0:  # 120 프레임마다 로그 (약 2초 분량)
            #     elapsed = time.time() - start_time
            #     fps = frames_count / elapsed if elapsed > 0 else 0
            #     logger.debug(f"애니메이션 데이터 전송 중: {frames_count}개 프레임, FPS: {fps:.1f}")
            
            if not out.first_frame_fut.done():
                out.first_frame_fut.set_result(None)
                logger.debug("첫 번째 애니메이션 프레임 전송 알림 완료")
    except Exception as e:
        logger.error(f"애니메이션 데이터 전송 중 오류 발생: {e}", exc_info=True)
        raise
    finally:
        if isinstance(stf_output, _ACloseable):
            await stf_output.aclose()
        
        duration = time.time() - start_time
        fps = frames_count / duration if duration > 0 else 0
        logger.info(f"애니메이션 데이터 전송 완료: {frames_count}개 프레임, 소요 시간: {duration:.2f}초, 평균 FPS: {fps:.1f}")
        animation_output.flush()

