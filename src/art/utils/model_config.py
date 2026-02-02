"""Model-specific configuration for chat templates and training defaults."""


def detect_chat_template_parts(
    tokenizer: object,
) -> tuple[str, str]:
    """Detect instruction and response parts from a tokenizer's chat template.

    Args:
        tokenizer: A tokenizer with a chat_template attribute

    Returns:
        Tuple of (instruction_part, response_part)

    Raises:
        ValueError: If the tokenizer has no chat_template or the format is unrecognized
    """
    if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
        raise ValueError(
            "Cannot detect chat template parts: tokenizer has no chat_template attribute. "
            "Please specify instruction_part and response_part manually."
        )

    template: str = tokenizer.chat_template

    # ChatML format (Qwen, etc.)
    if "<|im_start|>" in template:
        return "<|im_start|>user\n", "<|im_start|>assistant\n"

    # Llama 3 format
    if "<|start_header_id|>" in template:
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    # Gemma format
    if "<start_of_turn>" in template:
        return "<start_of_turn>user\n", "<start_of_turn>model\n"

    # Mistral format
    if "[INST]" in template:
        return "[INST]", "[/INST]"

    raise ValueError(
        f"Unrecognized chat template format. "
        f"Please specify instruction_part and response_part manually. "
        f"Template starts with: {template[:100]!r}..."
    )


def get_instruction_response_parts(
    model_id: str,
    tokenizer: object,
) -> tuple[str, str]:
    """Get instruction and response parts for a model by detecting from tokenizer.

    Args:
        model_id: The model identifier (used in error messages)
        tokenizer: Tokenizer with chat_template attribute

    Returns:
        Tuple of (instruction_part, response_part)

    Raises:
        ValueError: If chat template cannot be detected
    """
    try:
        return detect_chat_template_parts(tokenizer)
    except ValueError as e:
        raise ValueError(f"Failed to detect chat template for {model_id}: {e}") from e
