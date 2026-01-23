import tf_keras as tfk
from loguru import logger
from tf_keras import layers as tfkl


def define_classifier(ngens, seed=1, num_layers=1):

    if num_layers == 1:
        classifier = tfk.Sequential(
            [tfkl.InputLayer(input_shape=[ngens]), tfkl.BatchNormalization(), tfkl.Dense(1, activation="sigmoid")]
        )
    if num_layers == 2:
        logger.info("using 2 layers in classifier")
        classifier = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=[ngens]),
                tfkl.BatchNormalization(),
                tfkl.Dense(3, activation="relu"),
                tfkl.Dense(1, activation="sigmoid"),
            ]
        )

    model = tfk.Model(inputs=classifier.inputs, outputs=classifier.outputs[0])

    return model
