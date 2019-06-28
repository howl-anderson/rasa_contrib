# rasa_contrib

rasa_contrib is a addon package for [rasa](https://github.com/RasaHQ/rasa). It provide some useful/powerful addition component.

## component

Currently, it includes:

*  addons_intent_classifier_textcnn_tf

    TextCNN based intent classifier, based on TensorFlow
    
*  addons_intent_classifier_textcnn_paddle

    TextCNN based intent classifier, based on PaddlePaddle
    
*  addons_ner_bilstm_crf_tf

    Embedding+BiLSTM+CRF based NER extractor, based on TensorFlow
    
*  addons_ner_bilstm_crf_paddle

    Embedding+BiLSTM+CRF based NER extractor, based on PaddlePaddle


It also includes (but still work in progress):

*  MicroAddonsTokenizer

    Chinese tokenizer component, based on [MicroTokenizer](https://github.com/howl-anderson/MicroTokenizer)
    
*  StackedBilstmTensorFlowPolicy

    Stacked Bilstm based dialog policy, based on TensorFlow
    
*  StackedBilstmPaddlePolicy

    Stacked Bilstm based dialog policy, based on PaddlePaddle
    

## how to use it
Using the class path to the place where you should given a component name in config.yaml. This is a feature of rasa, see here for more document from rasa official document.

For example, your config.yml can be:
```yaml
language: "zh"

pipeline:
  - name: "rasa_contrib.nlu.utils.tensorflow_utils.TensorflowNLP"
  - name: "rasa_contrib.nlu.extractors.bilstm_crf_tf_entity_extractor.BilstmCrfTensorFlowEntityExtractor"
    max_steps: 600
  - name: "rasa_contrib.nlu.classifiers.text_cnn_tf_classifier.TextCnnTensorFlowClassifier"
    max_steps: 600

policies:
  - name: MemoizationPolicy
  - name: rasa_contrib.core.policies.stacked_bilstm_paddle_policy.StackedBilstmTensorFlowPolicy
```
