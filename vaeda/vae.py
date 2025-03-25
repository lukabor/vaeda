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
    
    # Define the prior for KL divergence regularization
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
                           reinterpreted_batch_ndims=1)
    
    # Build the model using Functional API
    # Input layer
    input_layer = tfk.Input(shape=[ngens])
    
    # Encoder layers
    x = tfkl.Dense(256, activation='relu')(input_layer)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None)(x)
    encoder_output = tfpl.IndependentNormal(enc_sze, 
                                           activity_regularizer=tfpl.KLDivergenceRegularizer(prior))(x)
    
    # Create the encoder model
    encoder = tfk.Model(inputs=input_layer, outputs=encoder_output, name='encoder')
    
    # Decoder layers
    x = tfkl.Dense(256, activation='relu')(encoder_output)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None)(x)
    decoder_output = tfpl.IndependentNormal(ngens)(x)
    
    # Cluster classifier
    x = tfkl.BatchNormalization()(encoder_output)
    classifier_output = tfkl.Dense(num_clust, activation='sigmoid')(x)
    
    # Create the full model
    vae = tfk.Model(inputs=input_layer, outputs=[decoder_output, classifier_output])
    
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
    
    # Define the prior for KL divergence regularization
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(enc_sze), scale=1),
                           reinterpreted_batch_ndims=1)
    
    # Build the model using Functional API
    # Input layer
    input_layer = tfk.Input(shape=[ngens])
    
    # Encoder layers
    x = tfkl.Dense(256, activation='relu')(input_layer)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(enc_sze), activation=None)(x)
    encoder_output = tfpl.IndependentNormal(enc_sze, 
                                           activity_regularizer=tfpl.KLDivergenceRegularizer(prior))(x)
    
    # Create the encoder model
    encoder = tfk.Model(inputs=input_layer, outputs=encoder_output, name='encoder')
    
    # Decoder layers
    x = tfkl.Dense(256, activation='relu')(encoder_output)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dropout(rate=0.3)(x)
    x = tfkl.Dense(tfpl.IndependentNormal.params_size(ngens), activation=None)(x)
    decoder_output = tfpl.IndependentNormal(ngens)(x)
    
    # Create the full model
    vae = tfk.Model(inputs=input_layer, outputs=decoder_output)
    
    # Define negative log-likelihood loss
    def nll(x, rv_x): 
        rec = rv_x.log_prob(x)
        return -tf.math.reduce_sum(rec, axis=-1) 
    
    # Compile the model
    vae.compile(optimizer=tf.optimizers.Adamax(learning_rate=1e-3),
                loss=nll)
    
    return vae, encoder
