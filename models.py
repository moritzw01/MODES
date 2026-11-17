
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


import tensorflow as tf
import keras


class CachedEncoder(keras.Model):
    def __init__(self, backbone, heads, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.heads = heads
        self.cache = {}  # Dict: sample_idx -> CPU-Tensor

    def clear_cache(self): 
        self.cache = {}

    def get_representation(self, inputs, idx=None):
        # run this before the taping --> no taping of the backbone 
        if idx is None:
            return self.backbone(inputs, training=False)

        if isinstance(idx, int):
            batch_size = int(tf.shape(tf.nest.flatten(inputs)[0])[0])
            idx_list = list(range(idx, idx + batch_size))
        else:
            idx_list = idx.numpy().tolist() if hasattr(idx, "numpy") else list(idx)

        reps = [None] * len(idx_list)
        missing_pos = []
        missing_ids = []

        for pos, sample_id in enumerate(idx_list):
            if sample_id in self.cache:
                reps[pos] = self.cache[sample_id]
            else:
                missing_pos.append(pos)
                missing_ids.append(sample_id)

        if missing_pos:
            # lambda is a way to extract the data from nested data structs like dicts etc
            missing_inputs = tf.nest.map_structure(lambda x: tf.gather(x, missing_pos), inputs)
            computed = self.backbone(missing_inputs, training=False)

            with tf.device('/CPU:0'):
                for pos, sample_id, rep in zip(missing_pos, missing_ids, computed):
                    rep_cpu = tf.identity(rep)
                    self.cache[sample_id] = rep_cpu
                    reps[pos] = rep_cpu

        return tf.stack(reps, axis=0)

    def call(self, inputs, idx=None, representation=None, training=False):
        # calling with the representation outside of the taping 
        if representation is None:
            representation = self.get_representation(inputs, idx=idx)

        output1, output2 = self.heads(representation, training=training)
        return [output1, output2, representation]



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