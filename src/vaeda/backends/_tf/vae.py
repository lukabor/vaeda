"""VAE model definition using TensorFlow Probability and tf_keras.

Ported from the upstream/kostkalab v0.1.x implementation (TFP
``IndependentNormal`` layers with a ``KLDivergenceRegularizer`` prior and a
Keras model trained via ``fit``), so the TensorFlow backend reproduces the
original vaeda numerics.
"""

from __future__ import annotations

import tensorflow as tf
import tf_keras as tfk
import tf_keras.layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl


def define_clust_vae(
    enc_sze: int,
    ngens: int,
    num_clust: int,
    LR: float = 1e-3,
    clust_weight: float = 10000,
) -> tfk.Model:
    """Build and compile the cluster-supervised VAE (TFP/Keras).

    The loss is the reconstruction NLL plus the TFP KL regularizer on the
    latent posterior, with ``clust_weight``-weighted categorical
    cross-entropy on the cluster head.
    """
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(enc_sze), scale=1), reinterpreted_batch_ndims=1
    )

    encoder = tfk.Sequential(
        [
            tfkl.InputLayer(input_shape=[ngens]),
            tfkl.Dense(256, activation="relu"),
            tfkl.BatchNormalization(),
            tfkl.Dropout(rate=0.3),
            tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),
            tfpl.IndependentNormal(
                enc_sze, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
            ),
        ],
        name="encoder",
    )

    decoder = tfk.Sequential(
        [
            tfkl.InputLayer(input_shape=[enc_sze]),
            tfkl.Dense(256, activation="relu"),
            tfkl.BatchNormalization(),
            tfkl.Dropout(rate=0.3),
            tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),
            tfpl.IndependentNormal(ngens),
        ],
        name="decoder",
    )

    clust_classifier = tfk.Sequential(
        [
            tfkl.InputLayer(input_shape=[enc_sze]),
            tfkl.BatchNormalization(),
            tfkl.Dense(num_clust, activation="sigmoid"),
        ],
        name="clust_classifier",
    )

    inpt = tfk.Input(shape=ngens)
    z = encoder(inpt)
    recon = decoder(z)
    clust_pred = clust_classifier(z)

    vae = tfk.Model(inputs=[inpt], outputs=[recon, clust_pred])

    def nll(x: tf.Tensor, rv_x: tfd.Distribution) -> tf.Tensor:
        return -tf.math.reduce_sum(rv_x.log_prob(x), axis=-1)

    vae.compile(
        optimizer=tfk.optimizers.Adamax(learning_rate=LR),
        loss=[nll, "categorical_crossentropy"],
        loss_weights=[1, clust_weight],
    )

    return vae
