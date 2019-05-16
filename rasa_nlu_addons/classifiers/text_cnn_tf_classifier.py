import logging
import os
import shutil
import tempfile
import typing
from typing import Any, Dict, Optional, Text

from rasa_nlu.components import Component
from rasa_nlu.config import InvalidConfigError, RasaNLUModelConfig
from rasa_nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu.config import RasaNLUModelConfig
    from rasa_nlu.training_data import TrainingData
    from rasa_nlu.model import Metadata
    from rasa_nlu.training_data import Message


class TextCnnTensorFlowClassifier(Component):
    name = "addons_intent_classifier_textcnn_tf"

    provides = ["intent", "intent_ranking"]

    requires = ["addons_tf_input_fn", "addons_tf_input_meta"]

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 model_dir=None) -> None:

        self.result_dir = None if 'result_dir' not in component_config else \
        component_config['result_dir']

        self.predict_fn = None
        self.model_dir = model_dir

        super(TextCnnTensorFlowClassifier, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        return ["tensorflow", "seq2label"]

    def train(self,
              training_data: TrainingData,
              config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        from seq2label.input import build_input_func
        from seq2label.model import Model

        raw_config = config.for_component(self.name)

        print(raw_config)

        if 'result_dir' not in raw_config:
            raw_config['result_dir'] = tempfile.mkdtemp()

        model = Model(raw_config)

        config = model.get_default_config()
        config.update(raw_config)

        # task_status = TaskStatus(config)

        # read data according configure
        train_data_generator_func = kwargs.get('addons_tf_input_fn')
        corpus_meta_data = kwargs.get('addons_tf_input_meta')

        config['tags_data'] = corpus_meta_data['label']
        config['num_classes'] = len(config['tags_data'])

        print('')

        # build model according configure

        # send START status to monitor system
        # task_status.send_status(task_status.START)

        # train and evaluate model
        train_input_func = build_input_func(train_data_generator_func, config)

        # train_iterator = train_input_func()
        # import tensorflow as tf
        # import sys
        #
        # with tf.Session() as sess:
        #     sess.run(tf.tables_initializer())
        #
        #     counter = 0
        #     while True:
        #         try:
        #             value = sess.run(train_iterator[0]['words'])
        #             counter += 1
        #             print(value)
        #             break
        #         except tf.errors.OutOfRangeError:
        #             break
        #
        # print(counter)
        # #
        # sys.exit(0)

        evaluate_result, export_results, final_saved_model = model.train_and_eval_then_save(
            train_input_func,
            None,
            config
        )

        # task_status.send_status(task_status.DONE)

        self.result_dir = final_saved_model

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs
             ):
        # type: (...) -> Component
        """Load this component from file.

        After a component has been trained, it will be persisted by
        calling `persist`. When the pipeline gets loaded again,
        this component needs to be able to restore itself.
        Components can rely on any context attributes that are
        created by :meth:`components.Component.pipeline_init`
        calls to components previous
        to this one."""
        if cached_component:
            return cached_component
        else:
            component_config = model_metadata.for_component(cls.name)
            return cls(component_config, model_dir)

    def process(self, message: Message, **kwargs: Any) -> None:
        from tensorflow.contrib import predictor
        from seq2label.input import to_fixed_len

        real_result_dir = os.path.join(self.model_dir, self.result_dir)
        print(real_result_dir)

        if not self.predict_fn:
            self.predict_fn = predictor.from_saved_model(real_result_dir)

        input_text = message.text

        input_feature = {
            'words': [to_fixed_len([i for i in input_text], 20, '<pad>')],
        }

        print(input_feature)

        predictions = self.predict_fn(input_feature)
        label = predictions['label'][0].decode()

        intent = {"name": label,
                  "confidence": 1}

        ranking = zip([i.decode() for i in predictions['label_mapping']], [float(i) for i in predictions['label_prob'][0]])
        intent_ranking = [{"name": name,
                           "confidence": score}
                          for name, score in ranking]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Returns the metadata necessary to load the model again."""

        print(model_dir)
        saved_model_dir = os.path.join(model_dir, self.name)

        print(saved_model_dir)
        print(self.result_dir)

        shutil.copytree(self.result_dir, saved_model_dir)

        return {'result_dir': self.name}
