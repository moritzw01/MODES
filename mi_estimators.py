

from keras import layers
import tensorflow as tf 

class CLUB(tf.keras.Model):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    """
        This class provides the CLUB estimation to I(X,Y)
        Method:
            call() :        provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    """
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        
        # p_mu outputs mean of q(Y|X)
        self.p_mu = tf.keras.Sequential([
            layers.Dense(hidden_size // 2, activation='relu', input_shape=(x_dim,)),
            layers.Dense(y_dim)
        ])
        
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = tf.keras.Sequential([
            layers.Dense(hidden_size // 2, activation='relu', input_shape=(x_dim,)),
            layers.Dense(y_dim, activation='tanh')
        ])

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def call(self, x_samples, y_samples, training=False): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 / 2.0 / tf.exp(logvar)
        
        prediction_1 = tf.expand_dims(mu, axis=1)          # shape [nsample, 1, dim]
        y_samples_1 = tf.expand_dims(y_samples, axis=0)    # shape [1, nsample, dim]

        # log of conditional probability of negative sample pairs
        negative = - tf.reduce_mean((y_samples_1 - prediction_1)**2, axis=1) / 2.0 / tf.exp(logvar)

        return tf.reduce_mean(tf.reduce_sum(positive, axis=-1) - tf.reduce_sum(negative, axis=-1))

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return tf.reduce_mean(
            tf.reduce_sum(-(mu - y_samples)**2 / tf.exp(logvar) - logvar, axis=1), 
            axis=0
        )
    
    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class InfoNCE(tf.keras.Model):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = tf.keras.Sequential([
            layers.Dense(hidden_size, activation='relu', input_shape=(x_dim + y_dim,)),
            layers.Dense(1, activation='softplus')
        ])
    
    def call(self, x_samples, y_samples, training=False):  # samples have shape [sample_size, dim]
        sample_size = tf.shape(y_samples)[0]
        # reshape because keras can not interpret the pairs as batch size per default 
        feat_dim = tf.shape(x_samples)[-1] + tf.shape(y_samples)[-1] 
        
        x_tile = tf.tile(tf.expand_dims(x_samples, axis=0), [sample_size, 1, 1])
        y_tile = tf.tile(tf.expand_dims(y_samples, axis=1), [1, sample_size, 1])

        T0 = self.F_func(tf.concat([x_samples, y_samples], axis=-1))

        # 3D -> 2D umformen für Keras 3 Sequential, danach wieder zurück zu 3D
        xy_tiled = tf.concat([x_tile, y_tile], axis=-1)
        xy_flat = tf.reshape(xy_tiled, [sample_size * sample_size, feat_dim])
        T1_flat = self.F_func(xy_flat)
        T1 = tf.reshape(T1_flat, [sample_size, sample_size, 1])

        sample_size_float = tf.cast(sample_size, tf.float32)
        lower_bound = tf.reduce_mean(T0) - (
            tf.reduce_mean(tf.reduce_logsumexp(T1, axis=1)) - tf.math.log(sample_size_float)
        )
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.call(x_samples, y_samples)

