from typing import Annotated, Literal

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pydantic
from pydantic import SkipValidation

Message = Annotated[ChatCompletionMessageParam, SkipValidation]
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = list[ChatCompletionToolParam]


class TrainConfig(pydantic.BaseModel):
    learning_rate: float = 5e-6
    beta: float = 0.0


class SFTConfig(pydantic.BaseModel):
    learning_rate: float = 2e-4
    batch_size: int | Literal["auto"] = "auto"
    custom_lr_schedule: list[float] = []


Verbosity = Literal[0, 1, 2]
