from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics


class VAE:
    def __init__(self, original_dim, latent_dim, intermediate_dim):
        """

        :param original_dim: Number of features in the original feature space
        :param latent_dim: Number of parameters you'd like to estimate
        :param intermediate_dim: Number of neurons in the hidden layer (should be less than original dim)
        """
        self.epsilon_std = 1.
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim

        # Define the encoding model
        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='relu')(x)

        # The parameters of the latent vector come from the same intermediate layer in the encoding model
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        # Model to go from latent Z vector back to original spatial dimension
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # instantiate VAE model
        self.vae = Model(x, x_decoded_mean)

        # Compute VAE loss
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='rmsprop')

        # build a model to project inputs on the latent space
        self.encoder = Model(x, z_mean)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

    def sampling(self, args):
        # Fancy math to transform the unit normal distribution to a parameterized Gaussian
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon