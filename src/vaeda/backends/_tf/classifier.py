"""Neural-network classifier for PU learning, using tf_keras.

Ported from the upstream/kostkalab v0.1.x implementation.
"""

from __future__ import annotations

import tf_keras as tfk
from loguru import logger
from tf_keras import layers as tfkl


def define_classifier(ngens: int, num_layers: int = 1) -> tfk.Model:
    """Build a binary classifier (1 or 2 dense layers) on batch-normed input."""
    if num_layers == 1:
        classifier = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=[ngens]),
                tfkl.BatchNormalization(),
                tfkl.Dense(1, activation="sigmoid"),
            ]
        )
    elif num_layers == 2:
        logger.info("using 2 layers in classifier")
        classifier = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=[ngens]),
                tfkl.BatchNormalization(),
                tfkl.Dense(3, activation="relu"),
                tfkl.Dense(1, activation="sigmoid"),
            ]
        )
    else:
        msg = "Only using 1 or 2 layers is supported"
        raise ValueError(msg)

    return tfk.Model(inputs=classifier.inputs, outputs=classifier.outputs[0])
