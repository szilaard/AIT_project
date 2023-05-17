from tensorflow.keras.layers import Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Embedding, Input
from tensorflow.keras import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from copy import deepcopy
from random import randint


class PositionwiseFeedForward(Layer):
    "Feedforward implementation"
    def __init__(self, d_model, d_ff=512, dropout=0.1, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        self.w_1 = Dense(d_ff, input_shape=(d_model,), activation='relu')
        self.w_2 = Dense(d_model, input_shape=(d_ff,), activation=None)
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        x = self.w_1(inputs)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff=512, dropout=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.size = d_model
        self.self_attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dff)
        self.sublayer1 = ResidualConnection(dropout)
        self.sublayer2 = ResidualConnection(dropout)
    
    def call(self, x, training=None, mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer2(x, self.feed_forward)
    
        
class ResidualConnection(Layer):
    "Residual connection implementation"
    def __init__(self, dropout=0.1, **kwargs):
        super(ResidualConnection, self).__init__(**kwargs)

        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout)

    def call(self, inputs, sublayer):
        x = self.layer_norm(inputs)
        x = sublayer(x)
        x = self.dropout(x)
        x = inputs + x
        return x
    
class PositionEmbeddingLayer(Layer):
    "Static sinusoidal positional embedding layer"
    def __init__(self, max_len, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        position_embedding_matrix = self.get_position_encoding(max_len, output_dim)
        self.position_embedding_layer = Embedding(
            input_dim=max_len,
            weights=[position_embedding_matrix],
            output_dim=output_dim
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_indices
    
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)
    
class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        for i, l in enumerate(self.enc_layers):
            l._name = "encoder_layer_{}".format(i)

        self.dropout = tf.keras.layers.Dropout(rate)


    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)