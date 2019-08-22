import logging
import os
import shutil
import tempfile
import typing
from typing import Any, Dict, Optional, Text

from rasa.nlu.components import Component
from rasa.nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message


class DenseNetworkTensorFlowClassifier(Component):
    name = "addons_intent_classifier_textcnn_tf"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 model_dir=None) -> None:

        self.result_dir = None if 'result_dir' not in component_config else \
        component_config['result_dir']

        self.predict_fn = None
        self.model_dir = model_dir

        super(DenseNetworkTensorFlowClassifier, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["tensorflow"]

    @staticmethod
    def build_model(feature_length, intent_number):
        import tensorflow as tf
        from tensorflow.keras import layers

        model = tf.keras.Sequential([
            # Adds a densely-connected layer with 64 units to the model:
            layers.Dense(64, activation='relu', input_shape=(feature_length,)),
            # Add another:
            layers.Dense(64, activation='relu'),
            # Add a softmax layer with 10 output units:
            layers.Dense(intent_number, activation='softmax')])

        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def get_input_data(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig):
        import numpy as np

        whole_intent_text_set = set()

        intent_text_list = []
        text_feature_list = []
        for example in training_data.training_examples:
            text_feature = example.get('text_features')
            text_feature_list.append(text_feature)

            intent_text = example.get('intent')
            intent_text_list.append(intent_text)
            whole_intent_text_set.add(intent_text)

        intent_lookup_table = {value: index for index, value in enumerate(whole_intent_text_set)}
        intent_int_list = [intent_lookup_table[i] for i in intent_text_list]

        intent_np_array = np.array(intent_int_list)
        text_feature_np_array = np.array(text_feature_list)

        intent_number = len(whole_intent_text_set)
        feature_length = text_feature_np_array.shape[-1]

        return text_feature_np_array, intent_np_array, feature_length, intent_number

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:
        import tensorflow as tf

        data, labels, feature_length, intent_number = self.get_input_data(training_data, config)

        model = self.build_model(feature_length, intent_number)

        model.fit(data, labels, epochs=10, batch_size=32)

        final_saved_model = './saved_model_dir'

        tf.keras.experimental.export_saved_model(model, final_saved_model)

        self.result_dir = final_saved_model

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any
    ) -> "Component":
        if cached_component:
            return cached_component
        else:
            return cls(meta, model_dir)

    def process(self, message: Message, **kwargs: Any) -> None:
        pass

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        print(model_dir)
        saved_model_dir = os.path.join(model_dir, self.name)

        print(saved_model_dir)
        print(self.result_dir)

        shutil.copytree(self.result_dir, saved_model_dir)

        return {'result_dir': self.name}
