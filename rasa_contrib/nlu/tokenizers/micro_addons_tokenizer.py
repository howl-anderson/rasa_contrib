import logging
import typing
from typing import Any, Dict, List, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    pass


class MicroAddonsTokenizer(Tokenizer):

    provides = ["tokens"]

    language_list = ["zh"]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        kwargs = copy.deepcopy(component_config)
        kwargs.pop("name")
        self.kwargs = kwargs
        super(MicroAddonsTokenizer, self).__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["MicroTokenizer"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        import MicroTokenizer

        text = message.get(attribute)

        tokenized = MicroTokenizer.cut(text, **self.kwargs)

        tokens = []
        offset = 0
        for word in tokenized:
            tokens.append(Token(word, offset))
            offset += len(word)

        return tokens
