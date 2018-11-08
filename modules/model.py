import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

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

    optimizer = tf.train.AdamOptimizer()
    def train(self, dataset):
        # initializing the hidden state at the start of every epoch
        hidden = self.reset_states()

        for (batch, (input, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # feeding the hidden state back into the model
                predictions, hidden = self(input, hidden)

                # reshape target to make loss function expect the target
                target = tf.reshape(target, (-1,))
                loss = self.loss_function(target, predictions)

            gradients = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(zip(gradients, self.variables))

            print("Batch: {}, Loss: {:.4f}".format(batch + 1, loss), end="\r")

        return loss.numpy()

    @staticmethod
    ## Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
    def loss_function(real, preds):
        return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

    def generate_text(self, dataset, start_string, gen_size=1000):
        generated_text = []
        # Vectorize start_string
        input_eval = tf.expand_dims(dataset.str_to_indices(start_string), 0)
        temperature = 1.0

        # hidden layer shape: (batch_size, units)
        hidden = [tf.zeros((1, self.units))]

        for i in tqdm(range(gen_size), desc="Generating..."):
            predictions, hidden = self(input_eval, hidden)

            # Using the multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

            # Pass the predicted word as the next input to the model along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            generated_text.append(predicted_id)

        return start_string + "".join(dataset.indices_to_str(generated_text)) + "\n"
