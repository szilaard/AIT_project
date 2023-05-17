from utils.layers import *
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAvgPool1D, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

def custom_binary_accuracy(y_true, y_pred, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_true.dtype)

    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)

def custom_binary_crossentropy(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    epsilon_ = K._constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    output = clip_ops.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = 4 * y_true * math_ops.log(output + K.epsilon())
    bce += (1 - y_true) * math_ops.log(1 - output + K.epsilon())
    return K.sum(-bce, axis=-1)

def transformer_classifier(
    num_layers=4,
    d_model=13,
    num_heads=3,
    dff=256,
    maximum_position_encoding=130,
    n_classes=10,
):
    inp = Input(shape=(None, d_model))

    encoder1 = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.3,
    )

    encoder2 = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.3,
    )

    encoder3 = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.3,
    )

    x = encoder1(inp)
    y = encoder2(inp)
    z = encoder3(inp)
    x = Dropout(0.2)(x)
    y = Dropout(0.2)(y)
    z = Dropout(0.2)(z)
    x = x + y + z
    x = tf.expand_dims(x, axis=-1)
    x = Conv2D(filters=128, kernel_size=3, activation="selu", padding='valid')(x)
    x = Conv2D(filters=64, kernel_size=3, activation="selu", padding='valid')(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(6 * n_classes, activation="selu")(x)
    x = Dropout(0.2)(x)
    out = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    opt = Adam(0.001)
    model.compile(
        optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']
        #optimizer=opt, loss=custom_binary_crossentropy, metrics=[custom_binary_accuracy]
    )
    model.summary()

    return model