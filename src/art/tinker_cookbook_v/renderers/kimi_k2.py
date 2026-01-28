"""Renderer for Moonshot AI's Kimi K2 models."""

import json
import re
import warnings

import tinker
import torch

from .base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    ensure_list,
    ensure_text,
    parse_response_for_stop_token,
    parse_think_blocks,
)

_TOOL_CALLS_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"
    r"|<\|tool_call_section_begin\|>(.*?)<\|tool_call_section_end\|>",
    re.DOTALL,
)
_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _split_tool_calls_section(content: str) -> tuple[str, str | None]:
    match = _TOOL_CALLS_SECTION_RE.search(content)
    if not match:
        return content, None
    tool_section = match.group(1) if match.group(1) is not None else match.group(2)
    return content[: match.start()], tool_section


def _extract_tool_name(tool_id: str) -> str:
    if not tool_id:
        return ""
    name_part = tool_id.split(":", 1)[0]
    if "." in name_part:
        _, name_part = name_part.split(".", 1)
    return name_part


def _parse_tool_calls_section(
    tool_section: str,
) -> tuple[list[ToolCall], list[UnparsedToolCall]]:
    tool_calls: list[ToolCall] = []
    unparsed_tool_calls: list[UnparsedToolCall] = []

    for match in _TOOL_CALL_RE.finditer(tool_section):
        raw_text = match.group(0)
        tool_id = match.group(1).strip()
        args_str = match.group(2).strip()
        func_name = _extract_tool_name(tool_id)

        try:
            json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    function=ToolCall.FunctionBody(name=func_name, arguments=args_str),
                    id=tool_id if tool_id else None,
                )
            )
        except json.JSONDecodeError as e:
            unparsed_tool_calls.append(
                UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")
            )

    return tool_calls, unparsed_tool_calls


class KimiK2Renderer(Renderer):
    """
    Format for moonshotai/Kimi-K2-Thinking:
        <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>
        <|im_user|>user<|im_middle|>What can you help me with?<|im_end|>
        <|im_assistant|>assistant<|im_middle|><think>reasoning</think>I can help you with...<|im_end|>

    Historical assistant messages use empty <think></think> blocks, while the final assistant
    response preserves reasoning_content in the thinking block.

    Note: Per the HuggingFace chat template, the default system message is automatically
    prepended if no system message is provided. This ensures train-eval consistency when
    using HF's apply_chat_template for inference.
    """

    DEFAULT_SYSTEM_PROMPT = "You are Kimi, an AI assistant created by Moonshot AI."

    def _ensure_system_message(self, messages: list[Message]) -> list[Message]:
        """Ensure a default system message is present if none exists.

        This matches the HuggingFace chat template behavior where a default system
        message is automatically added when none is provided.

        The default system message is inserted at the appropriate position:
        - If messages is empty: adds default system message
        - If starting with tool_declare: inserts default system after tool_declare (if no system message follows)
        - Otherwise: prepends default system message before first message (if first message isn't system)
        """
        if not messages:
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [default_system]

        # Accept both system and tool_declare as valid starting messages
        first_role = messages[0]["role"]
        if first_role == "tool_declare":
            # Check if a system message already exists after tool_declare
            if len(messages) >= 2 and messages[1]["role"] == "system":
                return messages
            # No system message, insert default after tool_declare
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [messages[0], default_system] + list(messages[1:])
        elif first_role != "system":
            default_system = Message(role="system", content=self.DEFAULT_SYSTEM_PROMPT)
            return [default_system] + list(messages)

        return messages

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """
        Render a message. For assistant messages, ctx.is_last controls whether thinking is preserved
        (True) or stripped to empty <think></think> (False).
        """
        role = message["role"]

        # Build role token based on role type
        if role == "user":
            header_str = f"<|im_user|>{role}<|im_middle|>"
        elif role == "assistant":
            header_str = f"<|im_assistant|>{role}<|im_middle|>"
        elif role == "system":
            header_str = f"<|im_system|>{role}<|im_middle|>"
        elif role == "tool_declare":
            # Tool declaration uses system token but with "tool_declare" as display name
            header_str = f"<|im_system|>{role}<|im_middle|>"
        elif role == "tool":
            # HF template uses message.name if present, otherwise role
            role_name = message.get("name")
            if not role_name:
                warnings.warn(
                    "Tool message missing 'name' field. Using 'tool' as fallback. "
                    "Consider setting 'name' to match the tool function name for better context.",
                    UserWarning,
                    stacklevel=3,
                )
                role_name = role
            header_str = f"<|im_system|>{role_name}<|im_middle|>"

            # Tool responses have special formatting - need tool_call_id to correlate with the call
            tool_call_id = message.get("tool_call_id", "")
            if not tool_call_id:
                warnings.warn(
                    "Tool message missing 'tool_call_id' field. KimiK2Renderer requires 'tool_call_id' "
                    "to render tool results correctly. The value should match ToolCall.id from the "
                    "assistant's tool_calls.",
                    UserWarning,
                    stacklevel=3,
                )
            header_str += f"## Return of {tool_call_id}\n"
        else:
            # Unknown roles default to system-style formatting
            header_str = f"<|im_system|>{role}<|im_middle|>"

        # Build output content
        output_str = ""
        if role == "assistant":
            # Extract thinking and text from content list
            parts = ensure_list(message["content"])
            thinking_content = "".join(
                p["thinking"] for p in parts if p["type"] == "thinking"
            )
            text_content = "".join(p["text"] for p in parts if p["type"] == "text")

            # For the last assistant message (is_last=True), preserve thinking; otherwise use empty think block
            if ctx.is_last and thinking_content:
                output_str = f"<think>{thinking_content}</think>"
            else:
                output_str = "<think></think>"
            output_str += text_content

            # Handle tool calls
            if "tool_calls" in message and message["tool_calls"]:
                output_str += "<|tool_calls_section_begin|>"
                for idx, tool_call in enumerate(message["tool_calls"]):
                    tool_id = tool_call.id
                    if not tool_id:
                        tool_id = f"functions.{tool_call.function.name}:{idx}"
                    args = tool_call.function.arguments
                    output_str += f"<|tool_call_begin|>{tool_id}<|tool_call_argument_begin|>{args}<|tool_call_end|>"
                output_str += "<|tool_calls_section_end|>"

        elif role == "tool_declare":
            # Tool declaration message: list tools with schema
            output_lines = ["You have access to the following tools:"]
            for tool in message.get("tools", []):
                output_lines.append(f"\n{tool['name']}: {tool.get('description', '')}")
                output_lines.append(json.dumps(tool["parameters"], ensure_ascii=False))
            output_str = "\n".join(output_lines)
        else:
            # System/user/tool messages use text content directly
            output_str = ensure_text(message["content"])

        output_str += "<|im_end|>"

        # Encode
        header_tokens = self.tokenizer.encode(header_str, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)

        return RenderedMessage(
            header=tinker.types.EncodedTextChunk(tokens=header_tokens),
            output=[tinker.types.EncodedTextChunk(tokens=output_tokens)],
        )

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._im_end_token
        )
        if not parse_success:
            return message, False

        assert isinstance(message["content"], str)
        content = message["content"]

        # Split tool calls section (if present)
        content, tool_section = _split_tool_calls_section(content)

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        unparsed_tool_calls: list[UnparsedToolCall] = []
        if tool_section is not None:
            tool_calls, unparsed_tool_calls = _parse_tool_calls_section(tool_section)
            if tool_calls:
                message["tool_calls"] = tool_calls
            if unparsed_tool_calls:
                message["unparsed_tool_calls"] = unparsed_tool_calls

        # Strip <think> blocks and parse structured content
        parts = parse_think_blocks(content)
        if parts is not None:
            message["content"] = parts
        else:
            message["content"] = content

        return message, True

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format."""
        result: dict = {"role": message["role"]}

        content = message["content"]
        if isinstance(content, str):
            result["content"] = content
        else:
            # Extract thinking into reasoning_content, keep text in content
            thinking_parts = []
            text_parts = []
            for p in content:
                if p["type"] == "thinking":
                    thinking_parts.append(p["thinking"])
                elif p["type"] == "text":
                    text_parts.append(p["text"])

            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["reasoning_content"] = "".join(thinking_parts)

        # Handle tool_calls
        if "tool_calls" in message and message["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message["tool_calls"]
            ]

        # Handle tool response fields
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]

        return result

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create tool declaration + optional system message."""
        messages: list[Message] = []
        if tools:
            messages.append(
                Message(
                    role="tool_declare",
                    content="",
                    tools=tools,  # type: ignore[typeddict-unknown-key]
                )
            )
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        return messages

    @property
    def _im_end_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._im_end_token]

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """
        Override to ensure default system prompt behavior aligns with HF template.
        """
        messages = self._ensure_system_message(messages)
        return super().build_supervised_example(messages, train_on_what)
