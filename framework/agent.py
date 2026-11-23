"""Agent framework for autonomous task execution with tool calling.

This module implements an agent that uses the OpenRouter API for LLM inference,
supporting streaming responses, tool calling, and reasoning token display.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from framework.llm import OpenRouterClient, OpenRouterConfig, TokenUsage

# Prefix that indicates the agent should stop (answer was submitted)
# This avoids global state - the tool result signals completion
ANSWER_SUBMITTED_PREFIX = "ANSWER_SUBMITTED:"

# How many times to remind the model to call a tool when it stops with plain text
MAX_SUBMIT_NUDGES = 3
# When this many iterations remain, warn the model to wrap up and submit
BUDGET_WARNING_REMAINING = 5
# In the last N iterations, only expose the finishing tool(s) so the model must submit
FORCE_FINISH_REMAINING = 3
FINISH_TOOL_NAMES = frozenset({"submit_answer"})

type ToolFunction = Callable[..., str]


class EventType(Enum):
    """Types of events emitted during agent execution."""

    # Generation events
    GENERATION_START = auto()
    THINKING_START = auto()
    THINKING_CHUNK = auto()
    THINKING_END = auto()
    RESPONSE_CHUNK = auto()
    GENERATION_END = auto()

    # Tool events
    TOOL_CALL_START = auto()
    TOOL_CALL_PARSED = auto()
    TOOL_EXECUTION_START = auto()
    TOOL_EXECUTION_END = auto()

    # Agent loop events
    ITERATION_START = auto()
    ITERATION_END = auto()
    AGENT_COMPLETE = auto()
    AGENT_ERROR = auto()


@dataclass
class AgentEvent:
    """An event emitted during agent execution."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format event for display."""
        return f"[{self.type.name}] {self.data}"


@dataclass
class Tool:
    """Represents a tool that can be called by the agent.

    Tool functions must return a string that will be shown to the LLM.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction


@dataclass
class ToolCall:
    """Represents a (parsed) tool call request from the agent."""

    id: str  # Required for OpenAI-compatible API
    name: str
    arguments: dict[str, Any]
    error: str | None = None


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str  # "system", "user", "assistant", or "tool"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None  # For assistant messages with tool calls
    tool_call_id: str | None = None  # For tool result messages


@dataclass
class ContextCompressionSettings:
    """Settings for context compression to reduce token usage."""

    enabled: bool = False
    keep_recent: int = 3  # Number of recent tool results to keep in full
    max_chars: int = 150  # Max chars for truncated older results


@dataclass
class Conversation:
    """Represents a conversation between the agent and the user."""

    messages: list[Message] = field(default_factory=list)

    def to_api_format(
        self,
        compression: ContextCompressionSettings | None = None,
    ) -> list[dict[str, Any]]:
        """Convert the conversation to OpenAI-compatible API format.

        Args:
            compression: Optional compression settings. If enabled, older tool
                results are truncated and duplicate consecutive tool calls are
                deduplicated.
        """
        messages_to_convert = self.messages

        if compression and compression.enabled:
            messages_to_convert = _compress_messages(
                self.messages,
                keep_recent=compression.keep_recent,
                max_chars=compression.max_chars,
            )

        result: list[dict[str, Any]] = []
        for message in messages_to_convert:
            msg: dict[str, Any] = {"role": message.role}

            if message.content is not None:
                msg["content"] = message.content

            if message.tool_calls is not None:
                msg["tool_calls"] = message.tool_calls

            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id

            result.append(msg)
        return result


def _truncate_tool_result(content: str, max_chars: int) -> str:
    """Truncate a tool result to max_chars with a summary prefix."""
    if len(content) <= max_chars:
        return content

    # Extract first line as summary (often contains row/column counts)
    first_line = content.split("\n")[0]
    if len(first_line) <= max_chars - 20:
        return f"[Truncated] {first_line}"

    return f"[Truncated] {content[:max_chars - 15]}..."


def _compress_messages(
    messages: list[Message],
    keep_recent: int,
    max_chars: int,
) -> list[Message]:
    """Compress messages by truncating old tool results and deduplicating.

    Applies two optimizations:
    1. Truncates tool results older than keep_recent to max_chars
    2. Removes duplicate consecutive tool calls with identical results
    """
    # Find all tool message indices (for determining which are "recent")
    tool_indices: list[int] = [
        i for i, m in enumerate(messages) if m.role == "tool"
    ]

    # Indices of tool messages to keep in full (the most recent ones)
    recent_tool_indices = set(tool_indices[-keep_recent:]) if tool_indices else set()

    # Build compressed message list
    result: list[Message] = []
    seen_tool_calls: dict[str, str] = {}  # (name, args_json) -> full result

    for i, msg in enumerate(messages):
        if msg.role == "tool":
            # Check for deduplication: same tool call with same result
            # Find the corresponding assistant message's tool call
            tool_key: str | None = None
            for j in range(i - 1, -1, -1):
                assistant_tool_calls = messages[j].tool_calls
                if messages[j].role == "assistant" and assistant_tool_calls:
                    for tc in assistant_tool_calls:
                        if tc.get("id") == msg.tool_call_id:
                            name = tc.get("function", {}).get("name", "")
                            args = tc.get("function", {}).get("arguments", "")
                            tool_key = f"{name}:{args}"
                            break
                    break

            # Deduplicate: if we've seen this exact call before with same result
            if tool_key and msg.content:
                if tool_key in seen_tool_calls:
                    prev_content = seen_tool_calls[tool_key]
                    if prev_content == msg.content:
                        # Skip this duplicate - but we need to keep the message
                        # structure for the API, so mark it as deduplicated
                        result.append(
                            Message(
                                role=msg.role,
                                content="[Duplicate call - see earlier result]",
                                tool_call_id=msg.tool_call_id,
                            )
                        )
                        continue
                seen_tool_calls[tool_key] = msg.content

            # Truncate if not in recent set
            if i not in recent_tool_indices and msg.content:
                result.append(
                    Message(
                        role=msg.role,
                        content=_truncate_tool_result(msg.content, max_chars),
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result


class Agent:
    """Implements a tiny, generic agent framework.

    Built on top of the OpenRouter API client.

    Only supports a single model, streaming, and an extensible tool set.
    """

    def __init__(self, config: OpenRouterConfig, tools: dict[str, Tool]):
        self.config = config
        self.tools: dict[str, Tool] = tools  # mapping from tool name to tool object
        self.client: OpenRouterClient = OpenRouterClient(config)
        self.conversation: Conversation = Conversation()
        self._compression = ContextCompressionSettings(
            enabled=config.compress_context,
            keep_recent=config.compress_keep_recent,
            max_chars=config.compress_max_chars,
        )
        self.reset_conversation()

    def _get_tool_definitions(self, finish_only: bool = False) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI-compatible format.

        With finish_only=True, only the finishing tools are exposed (hard step budget).
        """
        tools = [t for t in self.tools.values() if not finish_only or t.name in FINISH_TOOL_NAMES]
        if not tools:  # no finishing tool registered - fall back to all tools
            tools = list(self.tools.values())
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result as a string.

        Guaranteed to return a string, swallowing exceptions.
        """
        if tool_call.error:
            return f"Error parsing arguments for tool '{tool_call.name}': {tool_call.error}"

        if tool_call.name not in self.tools:
            return f"Error: Unknown tool '{tool_call.name}'"
        tool = self.tools[tool_call.name]
        try:
            return tool.function(**tool_call.arguments)
        except Exception as e:
            return f"Error executing {tool_call.name}: {e}"

    def _generate_response(
        self, conversation: Conversation, finish_only: bool = False
    ) -> Iterator[AgentEvent]:
        """Generate a response from the model, streaming the events out."""
        yield AgentEvent(type=EventType.GENERATION_START)

        messages = conversation.to_api_format(compression=self._compression)
        tools = self._get_tool_definitions(finish_only) if self.tools else None

        full_content = ""
        tool_calls: list[dict[str, Any]] = []
        in_thinking = False
        finish_reason: str | None = None
        usage: TokenUsage | None = None

        for chunk in self.client.chat_completion_stream(messages, tools):
            # Handle reasoning/thinking tokens
            if chunk.reasoning_details:
                for detail in chunk.reasoning_details:
                    if detail.get("type") == "reasoning.text":
                        text = detail.get("text", "")
                        if text:
                            if not in_thinking:
                                in_thinking = True
                                yield AgentEvent(type=EventType.THINKING_START)
                            yield AgentEvent(
                                type=EventType.THINKING_CHUNK,
                                data={"chunk": text},
                            )

            # Handle regular content
            if chunk.content:
                # Close thinking block if we were in it
                if in_thinking:
                    in_thinking = False
                    yield AgentEvent(type=EventType.THINKING_END)

                full_content += chunk.content
                yield AgentEvent(
                    type=EventType.RESPONSE_CHUNK,
                    data={"chunk": chunk.content},
                )

            # Handle tool calls (accumulated at the end)
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls

            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

            # Capture usage data (comes in final chunk)
            if chunk.usage:
                usage = chunk.usage

        # Close thinking if still open
        if in_thinking:
            yield AgentEvent(type=EventType.THINKING_END)

        event_data: dict[str, Any] = {
            "full_response": full_content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }
        if usage:
            event_data["usage"] = usage

        yield AgentEvent(type=EventType.GENERATION_END, data=event_data)

    def _get_system_message(self) -> str:
        """Get the system message for the agent (schema + guide index + workflow)."""
        from framework.prompts import build_system_prompt

        return build_system_prompt()

    def run(self, prompt: str) -> Iterator[AgentEvent]:
        """Run the agent with streaming output, from the user's natural language prompt."""
        # Add the new user message to the ongoing conversation
        self.conversation.messages.append(Message(role="user", content=prompt))

        # Track cumulative token usage across all iterations
        total_usage = TokenUsage()
        nudges = 0

        for iteration in range(self.config.max_iterations):
            yield AgentEvent(type=EventType.ITERATION_START, data={"iteration": iteration + 1})

            remaining = self.config.max_iterations - iteration
            if remaining == BUDGET_WARNING_REMAINING:
                # Budget warning: thorough (high-reasoning) runs otherwise explore until
                # they hit max_iterations and end with no answer at all.
                self.conversation.messages.append(
                    Message(
                        role="user",
                        content=(
                            f"Step budget: only {remaining} tool-call rounds remain. Stop "
                            "exploring. Finalize your best query now and call submit_answer "
                            "within the next 1-2 steps."
                        ),
                    )
                )

            full_response = ""
            tool_calls_data: list[dict[str, Any]] = []

            finish_only = remaining <= FORCE_FINISH_REMAINING
            if finish_only and remaining == FORCE_FINISH_REMAINING:
                self.conversation.messages.append(
                    Message(
                        role="user",
                        content=(
                            "Exploration tools are now disabled. Call submit_answer with "
                            "your best query immediately."
                        ),
                    )
                )
            for event in self._generate_response(self.conversation, finish_only):
                yield event
                if event.type == EventType.GENERATION_END:
                    full_response = event.data.get("full_response", "")
                    tool_calls_data = event.data.get("tool_calls", [])
                    # Accumulate token usage
                    if "usage" in event.data and event.data["usage"]:
                        total_usage = total_usage + event.data["usage"]

            # Parse tool calls from the structured response
            tool_calls = _parse_tool_calls_from_api(tool_calls_data)

            if not tool_calls:
                # The model stopped without calling a tool (e.g. it wrote the SQL as plain
                # text). Nudge it to use the tool instead of ending without an answer.
                if nudges < MAX_SUBMIT_NUDGES:
                    nudges += 1
                    self.conversation.messages.append(
                        Message(role="assistant", content=full_response or "")
                    )
                    self.conversation.messages.append(
                        Message(
                            role="user",
                            content=(
                                "You did not call a tool. Do not answer in plain text. "
                                "If you have verified your final query, call the "
                                "submit_answer TOOL now with it; otherwise continue "
                                "investigating with the tools."
                            ),
                        )
                    )
                    continue

                # Non-empty response without tool calls - agent is done
                self.conversation.messages.append(
                    Message(role="assistant", content=full_response)
                )
                yield AgentEvent(
                    type=EventType.AGENT_COMPLETE,
                    data={"response": full_response, "usage": total_usage},
                )
                return

            yield AgentEvent(type=EventType.TOOL_CALL_START, data={"count": len(tool_calls)})

            # Save assistant message with tool calls
            self.conversation.messages.append(
                Message(
                    role="assistant",
                    content=full_response if full_response else None,
                    tool_calls=tool_calls_data,
                )
            )

            for tool_call in tool_calls:
                yield AgentEvent(
                    type=EventType.TOOL_CALL_PARSED,
                    data={"name": tool_call.name, "arguments": tool_call.arguments},
                )
                yield AgentEvent(
                    type=EventType.TOOL_EXECUTION_START,
                    data={"name": tool_call.name},
                )
                tool_result = self._execute_tool(tool_call)
                yield AgentEvent(
                    type=EventType.TOOL_EXECUTION_END,
                    data={"name": tool_call.name, "result": tool_result},
                )

                # Add tool result message with tool_call_id
                self.conversation.messages.append(
                    Message(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_call.id,
                    )
                )

                # Check if this tool signals agent completion (e.g., answer submitted)
                if tool_result.startswith(ANSWER_SUBMITTED_PREFIX):
                    yield AgentEvent(
                        type=EventType.AGENT_COMPLETE,
                        data={
                            "reason": "answer_submitted",
                            "tool": tool_call.name,
                            "usage": total_usage,
                        },
                    )
                    return

            yield AgentEvent(type=EventType.ITERATION_END, data={"iteration": iteration + 1})

        yield AgentEvent(
            type=EventType.AGENT_ERROR,
            data={"error": "Max iterations reached", "usage": total_usage},
        )

    def reset_conversation(self) -> None:
        """Reset the conversation to the initial state (with system message)."""
        self.conversation = Conversation()
        self.conversation.messages.append(
            Message(role="system", content=self._get_system_message())
        )


def _parse_tool_calls_from_api(tool_calls_data: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse tool calls from OpenAI-compatible API response format."""
    tool_calls: list[ToolCall] = []

    for tc in tool_calls_data:
        tc_id = tc.get("id", "")
        function = tc.get("function", {})
        name = function.get("name", "")
        arguments_str = function.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
            error = None
        except json.JSONDecodeError as e:
            # Don't print to stdout, return error in ToolCall
            arguments = {}
            error = f"Invalid JSON arguments: {e}"

        tool_calls.append(
            ToolCall(
                id=tc_id,
                name=name,
                arguments=arguments,
                error=error,
            )
        )

    return tool_calls
