
import keras
import tensorflow as tf

class DINOv2Decoder(keras.Model):

    def __init__(
        self,
        latent_dim=256,
        image_size=224,
        channels=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert image_size == 224, "Currently supports 224x224"

        self.decoder = keras.Sequential([

            # Project latent vector into spatial representation
            keras.layers.Dense(7 * 7 * 256, activation="gelu"),
            keras.layers.Reshape((7, 7, 256)),

            # 7 -> 14
            keras.layers.Conv2DTranspose(
                256, 4, strides=2, padding="same",
                activation="gelu"
            ),

            # 14 -> 28
            keras.layers.Conv2DTranspose(
                128, 4, strides=2, padding="same",
                activation="gelu"
            ),

            # 28 -> 56
            keras.layers.Conv2DTranspose(
                64, 4, strides=2, padding="same",
                activation="gelu"
            ),

            # 56 -> 112
            keras.layers.Conv2DTranspose(
                32, 4, strides=2, padding="same",
                activation="gelu"
            ),

            # 112 -> 224
            keras.layers.Conv2DTranspose(
                channels, 4, strides=2, padding="same",
                activation="sigmoid"
            )

        ])

    def call(self, z, training=False):
        return self.decoder(z, training=training)


class DINOv2EmbeddingDecoder(keras.Model):
    def __init__(self, latent_dim=256, output_dim=768, **kwargs):
        super().__init__(**kwargs)

        self.decoder = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            keras.layers.Dense(512, activation="gelu"),
            keras.layers.LayerNormalization(),
            keras.layers.Dense(output_dim)
        ])

    def call(self, z, training=False):
        return self.decoder(z, training=training)


class BERTDecoder(keras.Model):

    def __init__(
        self,
        latent_dim=256,
        sequence_length=128,
        vocab_size=30522,
        hidden_dim=512,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.decoder = keras.Sequential([

            keras.layers.Dense(hidden_dim, activation="gelu"),
            keras.layers.LayerNormalization(),

            keras.layers.Dense(hidden_dim, activation="gelu"),

            # Reconstruct token distributions
            keras.layers.Dense(
                sequence_length * vocab_size
            ),

            keras.layers.Reshape(
                (sequence_length, vocab_size)
            )

        ])

    def call(self, z, training=False):
        return self.decoder(z, training=training)


class BERTEmbeddingDecoder(keras.Model):
    def __init__(self, latent_dim=256, output_dim=512, **kwargs):
        super().__init__(**kwargs)

        self.decoder = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            keras.layers.Dense(512, activation="gelu"),
            keras.layers.LayerNormalization(),
            keras.layers.Dense(output_dim)
        ])

    def call(self, z, training=False):
        return self.decoder(z, training=training)


class RepresentationAdapter(keras.layers.Layer):

    def call(self, inputs):
        raise NotImplementedError

class CLSAdapter(RepresentationAdapter):

    def call(self, inputs):
        return inputs[:, 0, :]


class MeanPoolingAdapter(RepresentationAdapter):

    def call(self, inputs):
        return tf.reduce_mean(inputs[:, 1:, :], axis=1)


class BertPooledAdapter(RepresentationAdapter):

    def call(self, inputs):
        return inputs["pooled_output"]


class BertMeanPoolingAdapter(RepresentationAdapter):

    def call(self, inputs):
        sequence = inputs["sequence_output"]
        mask = tf.cast(inputs["padding_mask"], sequence.dtype)
        mask = tf.expand_dims(mask, axis=-1)

        return tf.reduce_sum(sequence * mask, axis=1) / (
            tf.reduce_sum(mask, axis=1) + 1e-9
        )

# extracts the correct representation from the encoder 
def dino_cls(x):
    return x[:, 0, :]

def dino_mean(x):
    return tf.reduce_mean(x[:, 1:, :], axis=1)

def bert_cls(x):
    return x["sequence_output"][:, 0, :]

def bert_pooled(x):
    return x["pooled_output"]