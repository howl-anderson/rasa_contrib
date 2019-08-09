from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_contrib.nlu.featurizers.bert_base import BertBase

logger = logging.getLogger(__name__)


class BertTextFeaturizer(BertBase):
    provides = ["text_features"]

    def _set_feature(self, example, feature):
        new_feature = self._combine_with_existing_text_features(example, feature)

        example.set(
            "text_features",
            new_feature
        )
