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
    tfpl = tfp.layers
    tfd = tfp.distributions
    
    # Create a custom wrapper for TFP IndependentNormal
    class IndependentNormalLayer(tfkl.Layer):
        def __init__(self, event_shape, activity_regularizer=None, **kwargs):
            super().__init__(**kwargs)
            self.event_shape = event_shape
            self.activity_regularizer = activity_regularizer
            
        def build(self, input_shape):
            self._distribution_layer = tfpl.IndependentNormal(
                self.event_shape, 
                activity_regularizer=self.activity_regularizer
            )
            super().build(input_shape)
            
        def call(self, inputs, training=None):
            return self._distribution_layer(inputs)
            
        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.event_shape)
    
    # Define the prior for KL divergence regularization
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
                            reinterpreted_batch_ndims=1)
    
    # Build the encoder
    encoder_inputs = tfk.Input(shape=[ngens], name="encoder_input")
    x = tfkl.Dense(256, activation='relu')(encoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None)(x)
    encoder_outputs = IndependentNormalLayer(
        enc_sze, 
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior),
        name="encoder_distribution"
    )(x)
    
    # Create encoder model
    encoder = tfk.Model(inputs=encoder_inputs, outputs=encoder_outputs, name='encoder')
    
    # Build the decoder
    decoder_inputs = tfk.Input(shape=[enc_sze], name="decoder_input")
    x = tfkl.Dense(256, activation='relu')(decoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None)(x)
    decoder_outputs = IndependentNormalLayer(ngens, name="decoder_distribution")(x)
    
    # Create decoder model
    decoder = tfk.Model(inputs=decoder_inputs, outputs=decoder_outputs, name='decoder')
    
    # Build the classifier
    classifier_inputs = tfk.Input(shape=[enc_sze], name="classifier_input")
    x = tfkl.BatchNormalization()(classifier_inputs)
    classifier_outputs = tfkl.Dense(num_clust, activation='sigmoid')(x)
    
    # Create classifier model
    classifier = tfk.Model(inputs=classifier_inputs, outputs=classifier_outputs, name='classifier')
    
    # Connect all the components for the full VAE
    inputs = tfk.Input(shape=[ngens])
    z = encoder(inputs)
    outputs1 = decoder(z)
    outputs2 = classifier(z)
    
    # Create the full model
    vae = tfk.Model(inputs=inputs, outputs=[outputs1, outputs2])
    
    # Define negative log-likelihood loss
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    # Compile the model
    vae.compile(optimizer=tf.optimizers.Adamax(learning_rate=LR),
                loss=[nll, 'categorical_crossentropy'], 
                loss_weights=[1, clust_weight])
    
    return vae, encoder

def define_vae(enc_sze, ngens):
    # Convert dimensions to integers to avoid shape conversion issues
    enc_sze = int(enc_sze)
    ngens = int(ngens)
    
    tfk = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd = tfp.distributions
    
    # Create a custom wrapper for TFP IndependentNormal
    class IndependentNormalLayer(tfkl.Layer):
        def __init__(self, event_shape, activity_regularizer=None, **kwargs):
            super().__init__(**kwargs)
            self.event_shape = event_shape
            self.activity_regularizer = activity_regularizer
            
        def build(self, input_shape):
            self._distribution_layer = tfpl.IndependentNormal(
                self.event_shape, 
                activity_regularizer=self.activity_regularizer
            )
            super().build(input_shape)
            
        def call(self, inputs, training=None):
            return self._distribution_layer(inputs)
            
        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.event_shape)
    
    # Define the prior for KL divergence regularization
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
                            reinterpreted_batch_ndims=1)
    
    # Build the encoder
    encoder_inputs = tfk.Input(shape=[ngens], name="encoder_input")
    x = tfkl.Dense(256, activation='relu')(encoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None)(x)
    encoder_outputs = IndependentNormalLayer(
        enc_sze, 
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior),
        name="encoder_distribution"
    )(x)
    
    # Create encoder model
    encoder = tfk.Model(inputs=encoder_inputs, outputs=encoder_outputs, name='encoder')
    
    # Build the decoder
    decoder_inputs = tfk.Input(shape=[enc_sze], name="decoder_input")
    x = tfkl.Dense(256, activation='relu')(decoder_inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None)(x)
    decoder_outputs = IndependentNormalLayer(ngens, name="decoder_distribution")(x)
    
    # Create decoder model
    decoder = tfk.Model(inputs=decoder_inputs, outputs=decoder_outputs, name='decoder')
    
    # Connect the components for the full VAE
    inputs = tfk.Input(shape=[ngens])
    z = encoder(inputs)
    outputs = decoder(z)
    
    # Create the full model
    vae = tfk.Model(inputs=inputs, outputs=outputs)
    
    # Define negative log-likelihood loss
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    # Compile the model
    vae.compile(optimizer=tf.optimizers.Adamax(learning_rate=1e-3),
                loss=nll)
    
    return vae, encoder
