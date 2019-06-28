import logging
import typing
from typing import Any, Dict, List, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    pass


class MicroAddonsTokenizer(Tokenizer, Component):

    provides = ["tokens"]

    language_list = ["zh"]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        self.kwargs = component_config
        super(MicroAddonsTokenizer, self).__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["MicroTokenizer"]

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text: Text) -> List[Token]:
        import MicroTokenizer

        tokenized = MicroTokenizer.cut(text, **self.kwargs)

        tokens = []
        offset = 0
        for word in tokenized:
            tokens.append(Token(word, offset))
            offset += len(word)

        return tokens
