from typing import Iterable, Literal

import pydantic
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

Message = ChatCompletionMessageParam
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = list[ChatCompletionToolParam]


class TrainConfig(pydantic.BaseModel):
    learning_rate: float = 5e-6
    beta: float = 0.0


class SFTConfig(pydantic.BaseModel):
    learning_rate: float = 5e-5
    batch_size: int | Literal["auto"] = "auto"
    custom_lr_schedule: list[float] = []


Verbosity = Literal[0, 1, 2]
