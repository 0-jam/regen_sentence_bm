import tensorflow as tf
from tensorflow import keras

class Model(keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, force_cpu=False):
        super(Model, self).__init__()
        self.units = units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)

        # Enable CUDA if GPU is available
        if tf.test.is_gpu_available() and not force_cpu:
            self.gru = keras.layers.CuDNNGRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        else:
            self.gru = keras.layers.GRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform'
            )

        self.fc = keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        embedding = self.embedding(x)

        # output at any time step
        # output_shape = (batch_size, seq_length, hidden_size)
        # states_shape = (batch_size, hidden_size)
        output, states = self.gru(embedding, initial_state=hidden)

        # reshape the output to pass it to the Dense layer
        # after reshaping the shape is (batch_size * max_length, hidden)
        output = tf.reshape(output, (-1, output.shape[2]))

        # The dense layer will output predictons for every time_steps
        # output shape after the dense layer = (batch_size * seq_length, vocab_size)
        prediction = self.fc(output)

        return prediction, states
