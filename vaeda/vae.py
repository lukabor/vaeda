import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def define_clust_vae(enc_sze, ngens, num_clust, LR=1e-3, clust_weight=10000):
    # Convert dimensions to integers to avoid shape conversion issues
    enc_sze = int(enc_sze)
    ngens = int(ngens)
    num_clust = int(num_clust)
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions
    
    # Create a custom layer to wrap TFP layers
    class DistributionLayer(tfkl.Layer):
        def __init__(self, event_shape, regularizer=None, **kwargs):
            super().__init__(**kwargs)
            self.event_shape = event_shape
            self.regularizer = regularizer
            self.dist_layer = tfpl.IndependentNormal(
                event_shape, 
                activity_regularizer=regularizer
            )
            
        def call(self, inputs):
            return self.dist_layer(inputs)
            
        def compute_output_shape(self, input_shape):
            return input_shape[0], self.event_shape
    
    # Define the prior for KL divergence regularization
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
            reinterpreted_batch_ndims=1)
    
    # Build the encoder
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),
        DistributionLayer(enc_sze, regularizer=tfpl.KLDivergenceRegularizer(prior))
    ], name='encoder')
    
    # Build the decoder
    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),
        DistributionLayer(ngens)
    ], name='decoder')
    
    # Build the classifier
    clust_classifier = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.BatchNormalization(),
        tfkl.Dense(num_clust, activation='sigmoid')
    ], name='clust_classifier')
    
    # Connect the components
    IPT     = tfk.Input(shape=[ngens])
    z       = encoder(IPT)
    OPT1    = decoder(z)
    OPT2    = clust_classifier(z)
    vae = tfk.Model(inputs=[IPT],
                    outputs=[OPT1, OPT2])
    
    # Define negative log-likelihood loss
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    # Compile the model
    vae.compile(optimizer=tf.optimizers.Adamax(learning_rate=LR),
                loss=[nll, 'categorical_crossentropy'], 
                loss_weights=[1, clust_weight])
  
    return vae
    
def define_vae(enc_sze, ngens):
    # Convert dimensions to integers to avoid shape conversion issues
    enc_sze = int(enc_sze)
    ngens = int(ngens)
    
    tfk  = tf.keras
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd  = tfp.distributions
    
    # Create a custom layer to wrap TFP layers
    class DistributionLayer(tfkl.Layer):
        def __init__(self, event_shape, regularizer=None, **kwargs):
            super().__init__(**kwargs)
            self.event_shape = event_shape
            self.regularizer = regularizer
            self.dist_layer = tfpl.IndependentNormal(
                event_shape, 
                activity_regularizer=regularizer
            )
            
        def call(self, inputs):
            return self.dist_layer(inputs)
            
        def compute_output_shape(self, input_shape):
            return input_shape[0], self.event_shape
    
    # Define the prior for KL divergence regularization
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
            reinterpreted_batch_ndims=1)
    
    # Build the encoder
    encoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[ngens]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None),
        DistributionLayer(enc_sze, regularizer=tfpl.KLDivergenceRegularizer(prior))
    ], name='encoder')
    
    # Build the decoder
    decoder = tfk.Sequential([
        tfkl.InputLayer(input_shape=[enc_sze]),
        tfkl.Dense(256, activation='relu'),
        tfkl.BatchNormalization(),
        tfkl.Dropout(rate=0.3),
        tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None),
        DistributionLayer(ngens)
    ], name='decoder')
    
    # Connect the components
    IPT     = tfk.Input(shape=[ngens])
    z       = encoder(IPT)
    OPT1    = decoder(z)
    vae = tfk.Model(inputs=[IPT],
                    outputs=[OPT1])
    
    # Define negative log-likelihood loss
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    # Compile the model
    vae.compile(optimizer=tf.optimizers.Adamax(learning_rate=1e-3),
                loss=nll)
    
    return vae
