
import keras


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
