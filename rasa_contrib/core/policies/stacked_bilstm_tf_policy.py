import copy
import json
import logging
import os
import tensorflow as tf
import warnings
from typing import Any, List, Dict, Text, Optional, Tuple

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.featurizers import (
    MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer)
from rasa_core.featurizers import TrackerFeaturizer
from rasa_core.policies.policy import Policy
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class StackedBilstmTensorFlowPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    defaults = {
        # Neural Net and training params
        "rnn_size": 32,
        "epochs": 100,
        "batch_size": 32,
        "validation_split": 0.1
    }

    @staticmethod
    def _standard_featurizer(max_history=None):
        return MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                           max_history=max_history)

    def __init__(self,
                 featurizer: Optional[TrackerFeaturizer] = None,
                 model: Optional[tf.keras.models.Sequential] = None,
                 graph: Optional[tf.Graph] = None,
                 session: Optional[tf.Session] = None,
                 current_epoch: int = 0,
                 max_history: Optional[int] = None,
                 **kwargs: Any
                 ) -> None:
        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super(StackedBilstmTensorFlowPolicy, self).__init__(featurizer)

        self._load_params(**kwargs)
        self.model = model
        # by default keras uses default tf graph and global tf session
        # we are going to either load them or create them in train(...)
        self.graph = graph
        self.session = session

        self.current_epoch = current_epoch

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        self.rnn_size = config['rnn_size']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.validation_split = config['validation_split']

    @property
    def max_len(self):
        if self.model:
            return self.model.layers[0].batch_input_shape[1]
        else:
            return None

    def _build_model(self, num_features, num_actions, max_history_len):
        warnings.warn("Deprecated, use `model_architecture` instead.",
                      DeprecationWarning, stacklevel=2)
        return

    def model_architecture(
        self,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, Optional[int]]
    ) -> tf.keras.models.Sequential:
        """Build a keras model and return a compiled model."""

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Masking, LSTM, Dense, TimeDistributed, Activation)

        # Build Model
        model = Sequential()

        # the shape of the y vector of the labels,
        # determines which output from rnn will be used
        # to calculate the loss
        if len(output_shape) == 1:
            # y is (num examples, num features) so
            # only the last output from the rnn is used to
            # calculate the loss
            model.add(Masking(mask_value=-1, input_shape=input_shape))
            model.add(LSTM(self.rnn_size, dropout=0.2))
            model.add(Dense(input_dim=self.rnn_size, units=output_shape[-1]))
        elif len(output_shape) == 2:
            # y is (num examples, max_dialogue_len, num features) so
            # all the outputs from the rnn are used to
            # calculate the loss, therefore a sequence is returned and
            # time distributed layer is used

            # the first value in input_shape is max dialogue_len,
            # it is set to None, to allow dynamic_rnn creation
            # during prediction
            model.add(Masking(mask_value=-1,
                              input_shape=(None, input_shape[1])))
            model.add(LSTM(self.rnn_size, return_sequences=True, dropout=0.2))
            model.add(TimeDistributed(Dense(units=output_shape[-1])))
        else:
            raise ValueError("Cannot construct the model because"
                             "length of output_shape = {} "
                             "should be 1 or 2."
                             "".format(len(output_shape)))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        logger.debug(model.summary())

        return model

    def train(self,
              training_trackers: List[DialogueStateTracker],
              domain: Domain,
              **kwargs: Any
              ) -> None:

        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)

        # noinspection PyPep8Naming
        shuffled_X, shuffled_y = training_data.shuffled_X_y()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                if self.model is None:
                    self.model = self.model_architecture(shuffled_X.shape[1:],
                                                         shuffled_y.shape[1:])

                logger.info("Fitting model with {} total samples and a "
                            "validation split of {}"
                            "".format(training_data.num_examples(),
                                      self.validation_split))
                # filter out kwargs that cannot be passed to fit
                params = self._get_valid_params(self.model.fit, **kwargs)

                self.model.fit(shuffled_X, shuffled_y,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               **params)
                # the default parameter for epochs in keras fit is 1
                self.current_epoch = self.defaults.get("epochs", 1)
                logger.info("Done fitting keras policy model")

    def continue_training(self,
                          training_trackers: List[DialogueStateTracker],
                          domain: Domain,
                          **kwargs: Any) -> None:
        """Continues training an already trained policy."""

        # takes the new example labelled and learns it
        # via taking `epochs` samples of n_batch-1 parts of the training data,
        # inserting our new example and learning them. this means that we can
        # ask the network to fit the example without overemphasising
        # its importance (and therefore throwing off the biases)

        batch_size = kwargs.get('batch_size', 5)
        epochs = kwargs.get('epochs', 50)

        with self.graph.as_default(), self.session.as_default():
            for _ in range(epochs):
                training_data = self._training_data_for_continue_training(
                    batch_size, training_trackers, domain)

                # fit to one extra example using updated trackers
                self.model.fit(training_data.X, training_data.y,
                               epochs=self.current_epoch + 1,
                               batch_size=len(training_data.y),
                               verbose=0,
                               initial_epoch=self.current_epoch)

                self.current_epoch += 1

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:

        # noinspection PyPep8Naming
        X = self.featurizer.create_X([tracker], domain)

        with self.graph.as_default(), self.session.as_default():
            y_pred = self.model.predict(X, batch_size=1)

        if len(y_pred.shape) == 2:
            return y_pred[-1].tolist()
        elif len(y_pred.shape) == 3:
            return y_pred[0, -1].tolist()

    def persist(self, path: Text) -> None:

        if self.model:
            self.featurizer.persist(path)

            meta = {"model": "keras_model.h5",
                    "epochs": self.current_epoch}

            config_file = os.path.join(path, 'keras_policy.json')
            utils.dump_obj_as_json_to_file(config_file, meta)

            model_file = os.path.join(path, meta['model'])
            # makes sure the model directory exists
            utils.create_dir_for_file(model_file)
            with self.graph.as_default(), self.session.as_default():
                self.model.save(model_file, overwrite=True)
        else:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist then!")

    @classmethod
    def load(cls, path: Text) -> 'KerasPolicy':
        from tensorflow.keras.models import load_model

        if os.path.exists(path):
            featurizer = TrackerFeaturizer.load(path)
            meta_path = os.path.join(path, "keras_policy.json")
            if os.path.isfile(meta_path):
                meta = json.loads(utils.read_file(meta_path))

                model_file = os.path.join(path, meta["model"])

                graph = tf.Graph()
                with graph.as_default():
                    session = tf.Session()
                    with session.as_default():
                        model = load_model(model_file)

                return cls(featurizer=featurizer,
                           model=model,
                           graph=graph,
                           session=session,
                           current_epoch=meta["epochs"])
            else:
                return cls(featurizer=featurizer)
        else:
            raise Exception("Failed to load dialogue model. Path {} "
                            "doesn't exist".format(os.path.abspath(path)))
