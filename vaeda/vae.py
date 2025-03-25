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
    encoder_inputs = tfk.Input(shape=[ngens], name="encoder_input")
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
    
    # Sample from encoder distribution with explicit output shape function
    def sample_output_shape(input_shape):
        return (input_shape[0], enc_sze)
        
    latent_code = tfkl.Lambda(
        lambda dist: dist.sample(),
        output_shape=sample_output_shape,  # Explicitly define output shape
        name="latent_sample"
    )(encoder_dist)
    
    # Add KL loss with explicit output shape function
    def kl_output_shape(input_shape):
        return ()  # Scalar output
        
    kl_loss = tfkl.Lambda(
        lambda dist: tf.reduce_mean(
            tfd.kl_divergence(dist, prior_normal(enc_sze))
        ),
        output_shape=kl_output_shape,  # Explicitly define output shape
        name="kl_loss"
    )(encoder_dist)
    
    # Add KL loss as activity regularizer
    encoder_model = tfk.Model(
        inputs=encoder_inputs, 
        outputs=latent_code,
        name="encoder"
    )
    encoder_model.add_loss(kl_loss)
    
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
    latent = encoder_model(vae_inputs)
    reconstruction_dist = decoder_model(latent)
    classification = classifier_model(latent)
    
    # Create the full VAE model
    vae = tfk.Model(
        inputs=vae_inputs,
        outputs=[reconstruction_dist, classification],
        name="vae"
    )
    
    # Define negative log-likelihood loss
    def nll(x, rv_x):
        return -tf.reduce_sum(rv_x.log_prob(x), axis=-1)
    
    # Compile the model
    vae.compile(
        optimizer=tf.optimizers.Adamax(learning_rate=LR),
        loss=[nll, 'categorical_crossentropy'],
        loss_weights=[1, clust_weight]
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
    encoder_inputs = tfk.Input(shape=[ngens], name="encoder_input")
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
    
    # Sample from encoder distribution with explicit output shape function
    def sample_output_shape(input_shape):
        return (input_shape[0], enc_sze)
        
    latent_code = tfkl.Lambda(
        lambda dist: dist.sample(),
        output_shape=sample_output_shape,  # Explicitly define output shape
        name="latent_sample"
    )(encoder_dist)
    
    # Add KL loss with explicit output shape function
    def kl_output_shape(input_shape):
        return ()  # Scalar output
        
    kl_loss = tfkl.Lambda(
        lambda dist: tf.reduce_mean(
            tfd.kl_divergence(dist, prior_normal(enc_sze))
        ),
        output_shape=kl_output_shape,  # Explicitly define output shape
        name="kl_loss"
    )(encoder_dist)
    
    # Add KL loss as activity regularizer
    encoder_model = tfk.Model(
        inputs=encoder_inputs, 
        outputs=latent_code,
        name="encoder"
    )
    encoder_model.add_loss(kl_loss)
    
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
    latent = encoder_model(vae_inputs)
    reconstruction_dist = decoder_model(latent)
    
    # Create the full VAE model
    vae = tfk.Model(
        inputs=vae_inputs,
        outputs=reconstruction_dist,
        name="vae"
    )
    
    # Define negative log-likelihood loss
    def nll(x, rv_x):
        return -tf.reduce_sum(rv_x.log_prob(x), axis=-1)
    
    # Compile the model
    vae.compile(
        optimizer=tf.optimizers.Adamax(learning_rate=1e-3),
        loss=nll
    )
    
    return vae, encoder_model
