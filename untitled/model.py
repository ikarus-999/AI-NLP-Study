import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout
from keras.layers import Conv2D, MaxPooling2D, Embedding
from keras import regularizers

def CNN(vocab_size, sequence_length=30, n_class=2, embedding_size=300, filter_sizes=[3, 4, 5],
        num_filters=128, dropout_keep_prob=0.5, l2_reg_lambda=0.0, testmode=False):
    # input layer
    input_chars = Input(shape=(sequence_length,))

    # embedding
    embedded_chars = Embedding(vocab_size, embedding_size)(input_chars)
    embedded_chard_expanded = Reshape((sequence_length, embedding_size, 1))(embedded_chars)

    # conv-maxpool layers
    convs = []
    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, (filter_size, embedding_size), activation='relu')(embedded_chard_expanded)
        max_pool = MaxPooling2D((sequence_length - filter_size + 1, 1))(conv)
        convs.append(max_pool)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(convs, 3)
    h_pool_flat = tf.reshape(h_pool, (-1, num_filters_total))

    W = tf.Variable(name="W", shape=(num_filters_total, n_class),
                    initial_value=tf.initializers.GlorotUniform())
    b = tf.Variable(tf.constant(0.1, shape=(n_class, )), name="b")
    logits = tf.Variable(h_pool_flat * W + b, name="logits")
    prob = tf.reduce_max(tf.nn.softmax(logits), axis=1, name="prob")
    prediction = tf.cast(tf.argmax(logits, 1), tf.int32, name="predictions")

    merge = concatenate(convs)

    # dropout
    merge_dropout = Dropout(dropout_keep_prob)(merge)

    dense = Dense(n_class, activation='softmax', kernel_regularizer=regularizers.l2(l2_reg_lambda))(merge_dropout)
    model = Model(input_chars, dense)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    if testmode == True:
        return prob, prediction
    else:
        return model