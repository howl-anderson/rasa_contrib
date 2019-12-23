from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from typing import Any

import numpy as np

from rasa_contrib.nlu.featurizers.base_featurizer import ContribFeaturizer
from rasa.nlu.training_data import Message
from rasa_contrib.utils.trainning import BatchingIterator

logger = logging.getLogger(__name__)


class BertTextFeaturizer(ContribFeaturizer):
    provides = ["text_features"]

    defaults = {
        "ip": 'localhost',
        "port": 5555,
        "port_out": 5556,
        "show_server_config": False,
        "output_fmt": 'ndarray',
        "check_version": True,
        "timeout": 5000,
        "identity": None,
        "batch_size": 128
    }

    @classmethod
    def required_packages(cls):
        return ["bert_serving"]

    def __init__(self, component_config=None):
        super(BertTextFeaturizer, self).__init__(component_config)
        from bert_serving.client import ConcurrentBertClient

        self.bert_client = ConcurrentBertClient(
            ip=self.component_config['ip'],
            port=int(self.component_config['port']),
            port_out=int(self.component_config['port_out']),
            show_server_config=self.component_config['port_out'],
            output_fmt=self.component_config['output_fmt'],
            check_version=self.component_config['check_version'],
            timeout=int(self.component_config['timeout']),
            identity=self.component_config['identity']
        )

    def _query_embedding_vector(self, message_list):
        text_list = [i.text for i in message_list]

        embedding_vector_list = self.bert_client.encode(text_list,
                                                        is_tokenized=False)

        return np.squeeze(embedding_vector_list)

    def train(self, training_data, cfg=None, **kwargs):
        batch_iterator = BatchingIterator(self.component_config['batch_size'])

        for batch_examples in batch_iterator(training_data):
            embedding_vector_list = self._query_embedding_vector(
                batch_examples)

            for i, example in enumerate(batch_examples):
                example.set(
                    "text_features",
                    self._combine_with_existing_text_features(example,
                                                              embedding_vector_list[
                                                                  i])
                )

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        embedding_vector = self._query_embedding_vector([message])

        text_features = self._combine_with_existing_text_features(
            message,
            embedding_vector
        )

        message.set("text_features", text_features)
