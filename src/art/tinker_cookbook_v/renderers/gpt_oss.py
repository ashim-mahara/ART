"""GptOssRenderer - OpenAI's open source model format (Harmony)."""

from datetime import datetime
import json
import re
import warnings

import tinker
import torch

from ..tokenizer_utils import Tokenizer
from .base import (
    ContentPart,
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
    TrainOnWhat,
    UnparsedToolCall,
    ensure_list,
    ensure_text,
)

# =============================================================================
# TypeScript formatting utilities (stateless, used for Harmony tool definitions)
# =============================================================================


def _json_type_to_typescript(schema: dict) -> str:
    """Convert a single JSON schema type to TypeScript."""
    if "oneOf" in schema:
        return " | ".join(_json_type_to_typescript(s) for s in schema["oneOf"])
    if "anyOf" in schema:
        return " | ".join(_json_type_to_typescript(s) for s in schema["anyOf"])

    json_type = schema.get("type", "any")

    if isinstance(json_type, list):
        return " | ".join(_json_type_to_typescript({"type": t}) for t in json_type)

    if json_type == "string":
        if "enum" in schema:
            return " | ".join(json.dumps(v) for v in schema["enum"])
        base_type = "string"
    elif json_type == "number" or json_type == "integer":
        base_type = "number"
    elif json_type == "boolean":
        base_type = "boolean"
    elif json_type == "array":
        items_type = _json_type_to_typescript(schema.get("items", {}))
        base_type = f"{items_type}[]"
    elif json_type == "object":
        base_type = _json_schema_to_typescript(schema)
    else:
        base_type = "any"

    if schema.get("nullable"):
        return f"{base_type} | null"
    return base_type


def _json_schema_to_typescript(schema: dict) -> str:
    """Convert JSON schema to an inline TypeScript-ish type string."""
    if schema.get("type") != "object":
        return "any"

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    type_parts = []
    for prop_name, prop_schema in properties.items():
        prop_type = _json_type_to_typescript(prop_schema)
        optional = "" if prop_name in required else "?"
        type_parts.append(f"{prop_name}{optional}: {prop_type}")

    return "{ " + ", ".join(type_parts) + " }"


def _schema_comments(schema: dict) -> list[str]:
    """Extract comments from schema (title, description, examples)."""
    comments: list[str] = []
    title = schema.get("title")
    if title:
        comments.append(str(title))
        comments.append("")
    description = schema.get("description")
    if description:
        comments.append(str(description))
    examples = schema.get("examples")
    if examples:
        comments.append("Examples:")
        for example in examples:
            comments.append(f"- {json.dumps(example)}")
    return comments


def _format_parameters_block(schema: dict) -> str:
    """Format function parameters as a TypeScript-style block."""
    if schema.get("type") != "object" or not schema.get("properties"):
        return "()"

    lines = []
    header = "(_:"
    schema_description = schema.get("description")
    if schema_description:
        header += f" // {schema_description}"
    lines.append(header)
    lines.append("{")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for prop_name, prop_schema in properties.items():
        for comment in _schema_comments(prop_schema):
            lines.append(f"// {comment}")
        prop_type = _json_type_to_typescript(prop_schema)
        optional = "" if prop_name in required else "?"
        default_comment = ""
        if "default" in prop_schema:
            default_comment = f" // default: {json.dumps(prop_schema['default'])}"
        lines.append(f"{prop_name}{optional}: {prop_type},{default_comment}")

    lines.append("})")
    return "\n".join(lines)


def _format_tool_definition(tool: ToolSpec) -> str:
    """Format a single tool as a Harmony TypeScript-style definition."""
    lines = []
    if tool.get("description"):
        lines.append(f"// {tool['description']}")

    params = tool.get("parameters") or {}
    params_block = _format_parameters_block(params)
    lines.append(f"type {tool['name']} = {params_block} => any;")
    return "\n".join(lines)


class GptOssRenderer(Renderer):
    """
    Renderer for OpenAI's open source models using the Harmony format.

    Wire format: <|start|>role<|channel|>channel<|message|>content<|end|>
    No newlines between messages. Last assistant message ends with <|return|>;
    historical assistant messages end with <|end|>.

    Harmony Channels
    ----------------
    Each assistant message specifies a "channel" that controls how the content is
    interpreted and displayed. An assistant turn can have multiple channel segments
    (rendered as separate <|start|>assistant... blocks):

    - analysis: Chain-of-thought reasoning (hidden from end users, like <think> blocks)
    - commentary: Tool calls to developer-defined functions, or user-visible "preambles"
      before tool calls. Uses `to=functions.name` to route to specific tools.
    - final: The user-facing response text

    A typical assistant turn with thinking + tool call + final answer would render as:
        <|start|>assistant<|channel|>analysis<|message|>{thinking}<|end|>
        <|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{args}<|call|>
        ... (tool result) ...
        <|start|>assistant<|channel|>final<|message|>{answer}<|return|>

    Tool Calling
    ------------
    - Tool definitions: Go in developer message with TypeScript-style syntax
    - Tool calls: <|start|>assistant to=functions.name<|channel|>commentary <|constrain|>json<|message|>{args}<|call|>
    - Tool results: <|start|>functions.name to=assistant<|channel|>commentary<|message|>{result}<|end|>

    Reference: https://raw.githubusercontent.com/openai/openai-cookbook/main/articles/openai-harmony.md
    """

    # System prompt content (without rendering tokens). Tool channel instructions are NOT
    # included here; they are only added when tools are defined in the developer message.
    system_prompt_content = (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n"
        "Current date: {current_date}\n\n"
        "Reasoning: {reasoning_effort}\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    use_system_prompt: bool = False
    reasoning_effort: str | None = None
    current_date: str | None = (
        None  # If use_system_prompt=True, will use the current date if this is None. Set this to a fixed date for deterministic system prompt.
    )

    def __init__(
        self,
        tokenizer: Tokenizer,
        use_system_prompt: bool = False,
        reasoning_effort: str | None = None,
        current_date: str | None = None,
    ):
        super().__init__(tokenizer)
        self.use_system_prompt = use_system_prompt
        self.reasoning_effort = reasoning_effort
        self.current_date = current_date
        assert use_system_prompt == (reasoning_effort is not None), (
            "Reasoning effort must be set iff using system prompt"
        )

    # Internal role for OpenAI's system prompt (bypasses system->developer mapping)
    _INTERNAL_SYSTEM_ROLE = "_gptoss_internal_system"

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]

        # Handle tool result messages (role="tool")
        if role == "tool":
            return self._render_tool_result_message(message, ctx)

        # Internal system role renders as actual "system" without transformation
        if role == self._INTERNAL_SYSTEM_ROLE:
            role = "system"
        # User-provided "system" messages map to "developer" (per HF template)
        elif role == "system":
            role = "developer"

        header_str = f"<|start|>{role}"
        output_str = ""
        tool_calls: list[ToolCall] = []

        if message["role"] == "assistant":
            # Assistant channels. See https://cookbook.openai.com/articles/openai-harmony
            # Extract text and thinking from content list
            parts = ensure_list(message["content"])
            text_content = "".join(p["text"] for p in parts if p["type"] == "text")
            thinking_content = "".join(
                p["thinking"] for p in parts if p["type"] == "thinking"
            )
            tool_calls = message.get("tool_calls") or []

            # Analysis channel (CoT) - only if there's thinking content
            if thinking_content:
                output_str += f"<|channel|>analysis<|message|>{thinking_content}<|end|><|start|>assistant"

            # Commentary channel for tool calls
            if tool_calls:
                output_str += self._render_tool_calls(tool_calls)

            # Final channel for user-visible response
            if text_content or not tool_calls:
                output_str += f"<|channel|>final<|message|>{text_content}"
        else:
            # System/user/developer messages use "message" channel
            output_str = (
                f"<|channel|>message<|message|>{ensure_text(message['content'])}"
            )

        # End token depends on whether this is the last assistant message
        if message["role"] == "assistant" and ctx.is_last:
            output_str += "<|return|>"
        else:
            output_str += "<|end|>"

        # Build output chunks (single encoded text)
        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)
        output = [tinker.types.EncodedTextChunk(tokens=output_tokens)]

        return RenderedMessage(
            header=tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
            ),
            output=output,
        )

    def _render_tool_calls(self, tool_calls: list[ToolCall]) -> str:
        """
        Render tool calls in Harmony format.

        Format per tool call:
            <|start|>assistant to=functions.{tool_name}<|channel|>commentary <|constrain|>json<|message|>{args}<|call|>
        """
        tool_call_strs = []
        for tool_call in tool_calls:
            tool_call_strs.append(
                f"<|start|>assistant to=functions.{tool_call.function.name}<|channel|>commentary <|constrain|>json<|message|>{tool_call.function.arguments}<|call|>"
            )
        return "".join(tool_call_strs)

    def _render_tool_result_message(
        self, message: Message, ctx: RenderContext
    ) -> RenderedMessage:
        """Render tool result in Harmony format."""
        assert message["role"] == "tool"
        if "name" not in message:
            raise ValueError(
                "Tool result message must include 'name' field for Harmony"
            )

        header_str = f"<|start|>functions.{message['name']} to=assistant"
        output_str = (
            f"<|channel|>commentary<|message|>{ensure_text(message['content'])}<|end|>"
        )

        header_tokens = self.tokenizer.encode(header_str, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)

        return RenderedMessage(
            header=tinker.types.EncodedTextChunk(tokens=header_tokens),
            output=[tinker.types.EncodedTextChunk(tokens=output_tokens)],
        )

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse a Harmony-formatted response into a Message."""
        content = self.tokenizer.decode(response)

        # Parse tool calls (commentary channel with to=functions.name)
        tool_calls: list[ToolCall] = []
        unparsed_tool_calls: list[UnparsedToolCall] = []
        for match in re.finditer(
            r"<\|start\|>assistant to=functions\.(\w+)<\|channel\|>commentary <\|constrain\|>json<\|message\|>(.*?)<\|call\|>",
            content,
            re.DOTALL,
        ):
            raw_text = match.group(0)
            func_name, args_str = match.group(1), match.group(2).strip()
            try:
                json.loads(args_str)
                tool_calls.append(
                    ToolCall(
                        function=ToolCall.FunctionBody(
                            name=func_name, arguments=args_str
                        )
                    )
                )
            except json.JSONDecodeError as e:
                unparsed_tool_calls.append(
                    UnparsedToolCall(raw_text=raw_text, error=f"Invalid JSON: {e}")
                )

        # Extract assistant text content (final channel)
        text_match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>",
            content,
            re.DOTALL,
        )
        text_content = text_match.group(1) if text_match else ""

        # Extract thinking content (analysis channel)
        thinking_match = re.search(
            r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|><\|start\|>assistant",
            content,
            re.DOTALL,
        )
        thinking_content = thinking_match.group(1) if thinking_match else ""

        # Build structured content
        parts: list[ContentPart] = []
        if thinking_content:
            parts.append(ThinkingPart(type="thinking", thinking=thinking_content))
        if text_content:
            parts.append(TextPart(type="text", text=text_content))

        message = Message(role="assistant", content=parts)
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            message["unparsed_tool_calls"] = unparsed_tool_calls

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
        """Create developer/system messages for Harmony tool calling."""
        messages: list[Message] = []

        # Internal system prompt (if enabled)
        if self.use_system_prompt:
            current_date = self.current_date or datetime.utcnow().strftime("%Y-%m-%d")
            system_content = self.system_prompt_content.format(
                current_date=current_date, reasoning_effort=self.reasoning_effort
            )
            messages.append(
                Message(role=self._INTERNAL_SYSTEM_ROLE, content=system_content)
            )

        # Tool definitions go in developer message
        if tools:
            tool_defs = "\n\n".join(_format_tool_definition(t) for t in tools)
            tools_str = f"""You are given a set of tools.\n\n{tool_defs}"""
            messages.append(Message(role="developer", content=tools_str))

        # Optional user-provided system prompt (mapped to developer role)
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        return messages

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """
        Harmony needs special handling: render_message emits multiple assistant blocks
        per message (analysis/commentary/final), so simple concatenation does not align
        with generation prompt. We override to preserve these semantics and still apply
        token weights based on train_on_what.
        """
        if train_on_what == TrainOnWhat.ALL_ASSISTANT_MESSAGES:
            warnings.warn(
                "GptOssRenderer does not satisfy extension property; "
                "ALL_ASSISTANT_MESSAGES may lead to mismatched prefixes."
            )

        # Build tokens and weights by rendering each message and applying weights to outputs
        model_input_chunks_weights: list[
            tuple[tinker.types.ModelInputChunk, float]
        ] = []

        if self._bos_tokens:
            model_input_chunks_weights.append(
                (tinker.types.EncodedTextChunk(tokens=self._bos_tokens), 0.0)
            )

        for idx, message in enumerate(messages):
            ctx = RenderContext(
                idx=idx,
                is_last=(idx == len(messages) - 1),
                prev_message=messages[idx - 1] if idx > 0 else None,
            )
            rendered = self.render_message(message, ctx)

            # Header never trainable unless ALL_TOKENS
            header_weight = int(train_on_what == TrainOnWhat.ALL_TOKENS)
            if rendered.header:
                model_input_chunks_weights.append((rendered.header, header_weight))

            # Determine if this message's output should be weighted
            is_last_message = idx == len(messages) - 1
            is_assistant = message["role"] == "assistant"
            is_user_or_system = message["role"] in ["user", "system"]

            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    output_has_weight = is_last_message and is_assistant
                case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                    output_has_weight = is_assistant
                case TrainOnWhat.ALL_MESSAGES:
                    output_has_weight = True
                case TrainOnWhat.ALL_TOKENS:
                    output_has_weight = True
                case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                    output_has_weight = is_user_or_system
                case TrainOnWhat.CUSTOMIZED:
                    output_has_weight = message.get("trainable", False)
                case _:
                    raise ValueError(f"Unknown train_on_what: {train_on_what}")

            for output_part in rendered.output:
                if output_part:
                    model_input_chunks_weights.append(
                        (output_part, int(output_has_weight))
                    )

        weights_data = [
            w for chunk, w in model_input_chunks_weights for _ in range(chunk.length)
        ]
        weights_tensor = torch.tensor(weights_data)
        model_input_chunks = [chunk for chunk, _ in model_input_chunks_weights]
        return tinker.ModelInput(chunks=model_input_chunks), weights_tensor
