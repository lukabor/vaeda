import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def define_clust_vae(enc_sze, ngens, num_clust, LR=1e-3, clust_weight=10000):
    # Convert dimensions to integers to avoid shape conversion issues
    enc_sze = int(enc_sze)
    ngens = int(ngens)
    num_clust = int(num_clust)
    
    tfk = tf.keras
    tfkl = tf.keras.layers
    tfd = tfp.distributions
    
    # Define a custom distribution layer that properly handles TFP distributions
    class DistributionLayer(tfkl.Layer):
        def __init__(self, event_size, make_distribution_fn, **kwargs):
            super().__init__(**kwargs)
            self.event_size = event_size
            self.make_distribution_fn = make_distribution_fn
            self.params_size = 2 * event_size  # mean + log_scale

        def compute_output_shape(self, input_shape):
            return input_shape[:-1] + (self.event_size,)

        def build(self, input_shape):
            self.built = True

        def call(self, inputs):
            loc = inputs[..., :self.event_size]
            scale = tf.nn.softplus(inputs[..., self.event_size:]) + 1e-5
            return self.make_distribution_fn(loc=loc, scale=scale)
    
    # Standard normal prior
    def prior_normal(event_size):
        return tfd.Independent(
            tfd.Normal(loc=tf.zeros(event_size), scale=tf.ones(event_size)),
            reinterpreted_batch_ndims=1
        )
    
    # Create distribution functions
    def encoder_dist_fn(loc, scale):
        dist = tfd.Independent(
            tfd.Normal(loc=loc, scale=scale),
            reinterpreted_batch_ndims=1
        )
        return dist
    
    def decoder_dist_fn(loc, scale):
        dist = tfd.Independent(
            tfd.Normal(loc=loc, scale=scale),
            reinterpreted_batch_ndims=1
        )
        return dist
    
    # Build the encoder
    encoder_inputs = tfkl.Input(shape=[ngens], name="encoder_input")
    x = tfkl.Dense(256, activation='relu')(encoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(2 * enc_sze)(x)  # mean and log_var for each latent dimension
    
    # Create encoder distribution
    encoder_dist = DistributionLayer(
        event_size=enc_sze,
        make_distribution_fn=encoder_dist_fn,
        name="encoder_dist"
    )(x)
    
    # Sample from encoder distribution
    latent_code = encoder_dist.sample()
    
    # Calculate KL divergence from prior
    prior = prior_normal(enc_sze)
    kl_divergence = tfd.kl_divergence(encoder_dist, prior)
    kl_loss = tf.reduce_mean(kl_divergence)
    
    # Build standalone encoder model for later use
    encoder_model = tfk.Model(
        inputs=encoder_inputs, 
        outputs=latent_code,
        name="encoder"
    )
    
    # Build the decoder
    decoder_inputs = tfkl.Input(shape=[enc_sze], name="decoder_input")
    x = tfkl.Dense(256, activation='relu')(decoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(2 * ngens)(x)  # mean and log_var for output
    
    # Create decoder distribution
    decoder_dist = DistributionLayer(
        event_size=ngens,
        make_distribution_fn=decoder_dist_fn,
        name="decoder_dist"
    )(x)
    
    # Create decoder model
    decoder_model = tfk.Model(
        inputs=decoder_inputs,
        outputs=decoder_dist,
        name="decoder"
    )
    
    # Build the classifier
    classifier_inputs = tfk.Input(shape=[enc_sze], name="classifier_input")
    x = tfkl.BatchNormalization()(classifier_inputs)
    classifier_outputs = tfkl.Dense(num_clust, activation='sigmoid')(x)
    
    # Create classifier model
    classifier_model = tfk.Model(
        inputs=classifier_inputs,
        outputs=classifier_outputs,
        name="classifier"
    )
    
    # Build the full VAE
    vae_inputs = tfk.Input(shape=[ngens])
    # Use encoder without the Lambda layers to avoid shape inference issues
    x = tfkl.Dense(256, activation='relu')(vae_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(2 * enc_sze)(x)
    encoder_dist_vae = DistributionLayer(
        event_size=enc_sze,
        make_distribution_fn=encoder_dist_fn,
    )(x)
    latent = encoder_dist_vae.sample()
    
    # Calculate KL divergence directly in the model
    prior = prior_normal(enc_sze)
    kl_divergence = tfd.kl_divergence(encoder_dist_vae, prior)
    kl_loss_tensor = tf.reduce_mean(kl_divergence)
    
    # Convert kl_loss to tensor with shape [batch_size, 1]
    kl_loss_per_example = tf.ones_like(latent[:, 0:1]) * kl_loss_tensor / tf.cast(tf.shape(latent)[0], tf.float32)
    
    reconstruction_dist = decoder_model(latent)
    classification = classifier_model(latent)
    
    # Create the full VAE model - now include KL loss as an output
    vae = tfk.Model(
        inputs=vae_inputs,
        outputs=[reconstruction_dist, classification, kl_loss_per_example],
        name="vae"
    )
    
    # Define negative log-likelihood loss
    def nll(x, rv_x):
        return -tf.reduce_sum(rv_x.log_prob(x), axis=-1)
    
    # Custom loss for KL divergence (just return the pre-calculated value)
    def kl_loss_fn(_, y_pred):
        return y_pred
    
    # Compile the model with the KL loss included as a separate output
    vae.compile(
        optimizer=tf.optimizers.Adamax(learning_rate=LR),
        loss=[nll, 'categorical_crossentropy', kl_loss_fn],
        loss_weights=[1, clust_weight, 1]
    )
    
    return vae, encoder_model

def define_vae(enc_sze, ngens):
    # Convert dimensions to integers to avoid shape conversion issues
    enc_sze = int(enc_sze)
    ngens = int(ngens)
    
    tfk = tf.keras
    tfkl = tf.keras.layers
    tfd = tfp.distributions
    
    # Define a custom distribution layer that properly handles TFP distributions
    class DistributionLayer(tfkl.Layer):
        def __init__(self, event_size, make_distribution_fn, **kwargs):
            super().__init__(**kwargs)
            self.event_size = event_size
            self.make_distribution_fn = make_distribution_fn
            self.params_size = 2 * event_size  # mean + log_scale

        def compute_output_shape(self, input_shape):
            return input_shape[:-1] + (self.event_size,)

        def build(self, input_shape):
            self.built = True

        def call(self, inputs):
            loc = inputs[..., :self.event_size]
            scale = tf.nn.softplus(inputs[..., self.event_size:]) + 1e-5
            return self.make_distribution_fn(loc=loc, scale=scale)
    
    # Standard normal prior
    def prior_normal(event_size):
        return tfd.Independent(
            tfd.Normal(loc=tf.zeros(event_size), scale=tf.ones(event_size)),
            reinterpreted_batch_ndims=1
        )
    
    # Create distribution functions
    def encoder_dist_fn(loc, scale):
        dist = tfd.Independent(
            tfd.Normal(loc=loc, scale=scale),
            reinterpreted_batch_ndims=1
        )
        return dist
    
    def decoder_dist_fn(loc, scale):
        dist = tfd.Independent(
            tfd.Normal(loc=loc, scale=scale),
            reinterpreted_batch_ndims=1
        )
        return dist
    
    # Build the encoder
    encoder_inputs = tfkl.Input(shape=[ngens], name="encoder_input")
    x = tfkl.Dense(256, activation='relu')(encoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(2 * enc_sze)(x)  # mean and log_var for each latent dimension
    
    # Create encoder distribution
    encoder_dist = DistributionLayer(
        event_size=enc_sze,
        make_distribution_fn=encoder_dist_fn,
        name="encoder_dist"
    )(x)
    
    # Sample from encoder distribution
    latent_code = encoder_dist.sample()
    
    # Build standalone encoder model for later use
    encoder_model = tfk.Model(
        inputs=encoder_inputs, 
        outputs=latent_code,
        name="encoder"
    )
    
    # Build the decoder
    decoder_inputs = tfk.Input(shape=[enc_sze], name="decoder_input")
    x = tfkl.Dense(256, activation='relu')(decoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(2 * ngens)(x)  # mean and log_var for output
    
    # Create decoder distribution
    decoder_dist = DistributionLayer(
        event_size=ngens,
        make_distribution_fn=decoder_dist_fn,
        name="decoder_dist"
    )(x)
    
    # Create decoder model
    decoder_model = tfk.Model(
        inputs=decoder_inputs,
        outputs=decoder_dist,
        name="decoder"
    )
    
    # Build the full VAE
    vae_inputs = tfk.Input(shape=[ngens])
    # Use encoder without the Lambda layers to avoid shape inference issues
    x = tfkl.Dense(256, activation='relu')(vae_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(2 * enc_sze)(x)
    encoder_dist_vae = DistributionLayer(
        event_size=enc_sze,
        make_distribution_fn=encoder_dist_fn,
    )(x)
    latent = encoder_dist_vae.sample()
    
    # Calculate KL divergence directly in the model
    prior = prior_normal(enc_sze)
    kl_divergence = tfd.kl_divergence(encoder_dist_vae, prior)
    kl_loss_tensor = tf.reduce_mean(kl_divergence)
    
    # Convert kl_loss to tensor with shape [batch_size, 1]
    kl_loss_per_example = tf.ones_like(latent[:, 0:1]) * kl_loss_tensor / tf.cast(tf.shape(latent)[0], tf.float32)
    
    reconstruction_dist = decoder_model(latent)
    
    # Create the full VAE model - now include KL loss as an output
    vae = tfk.Model(
        inputs=vae_inputs,
        outputs=[reconstruction_dist, kl_loss_per_example],
        name="vae"
    )
    
    # Define negative log-likelihood loss
    def nll(x, rv_x):
        return -tf.reduce_sum(rv_x.log_prob(x), axis=-1)
    
    # Custom loss for KL divergence (just return the pre-calculated value)
    def kl_loss_fn(_, y_pred):
        return y_pred
    
    # Compile the model with the KL loss included as a separate output
    vae.compile(
        optimizer=tf.optimizers.Adamax(learning_rate=1e-3),
        loss=[nll, kl_loss_fn],
        loss_weights=[1, 1]
    )
    
    return vae, encoder_model
