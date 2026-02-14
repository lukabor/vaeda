import tf_keras as tfk
from loguru import logger
from tf_keras import layers as tfkl


def define_classifier(ngens, seed: int = 1, num_layers: int = 1):
    match num_layers:
        case 1:
            classifier = tfk.Sequential([
                tfkl.InputLayer(input_shape=[ngens]),
                tfkl.BatchNormalization(),
                tfkl.Dense(1, activation="sigmoid"),
            ])
        case 2:
            logger.info("using 2 layers in classifier")
            classifier = tfk.Sequential([
                tfkl.InputLayer(input_shape=[ngens]),
                tfkl.BatchNormalization(),
                tfkl.Dense(3, activation="relu"),
                tfkl.Dense(1, activation="sigmoid"),
            ])
        case _:
            msg = "Only using 1 or 2 layers is supported"
            raise ValueError(msg)

    model = tfk.Model(inputs=classifier.inputs, outputs=classifier.outputs[0])

    return model
