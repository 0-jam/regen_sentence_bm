import functools
import json
import pickle
import time
from pathlib import Path
from random import choice

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from modules.wakachi.mecab import divide_word

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

SEQ_LENGTH = 100
BUFFER_SIZE = 10000
NUM_HIDDEN_LAYERS = 1
WORD_LIMIT = 15000


def divide_text(text):
    sentences = []

    for line in text.split('\n'):
        sentences += divide_word(line) + ['\n']

    return sentences


class TextModel(object):
    def __init__(self):
        self.dataset = None
        self.trainer = None
        self.generator = None

    # Set hyper parameters from arguments
    def set_parameters(self, embedding_dim=256, units=1024, batch_size=64, cpu_mode=False):
        self.embedding_dim, self.units, self.batch_size, self.cpu_mode = embedding_dim, units, batch_size, cpu_mode

    # Preparing the dataset
    def build_dataset(self, text_path, char_level=True, encoding='utf-8'):
        self.tokenizer = keras.preprocessing.text.Tokenizer(filters='\\\t', oov_token='<oov>', char_level=char_level, num_words=WORD_LIMIT)

        with Path(text_path).open(encoding=encoding) as data:
            text = data.read()

        if not char_level:
            text = divide_text(text)

        # Vectorize the text
        self.tokenizer.fit_on_texts(text)
        # Index 0 is preserved in the Keras tokenizer for the unknown word, but it's not included in vocab2idx
        self.idx2vocab = {i: v for v, i in self.tokenizer.word_index.items()}
        # self.idx2vocab[0] = '<oov>'
        self.vocab_size = len(self.idx2vocab) + 1
        text_size = len(text)
        print("Text has {} characters ({} unique characters)".format(text_size, self.vocab_size - 1))

        # Creating a mapping from unique characters to indices
        text_as_int = self.vocab_to_indices(text)

        # The maximum length sentence we want for single input in characters
        chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(SEQ_LENGTH + 1, drop_remainder=True)
        self.dataset = chunks.map(self.split_into_target).shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True)
        self.steps_per_epoch = text_size // SEQ_LENGTH // self.batch_size

    # Save/Load the tokenizer as pickle
    def save_tokenizer(self, save_dir):
        with save_dir.joinpath('tokenizer.pickle').open('wb') as tokenizer_fp:
            pickle.dump(self.tokenizer, tokenizer_fp)

    def load_tokenizer(self, load_dir):
        with Path(load_dir).joinpath('tokenizer.pickle').open('rb') as tokenizer_fp:
            self.tokenizer = pickle.load(tokenizer_fp)

        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.idx2vocab = {i: v for v, i in self.tokenizer.word_index.items()}

    def is_word_based(self):
        return not self.tokenizer.char_level

    # Return model settings as dict
    def parameters(self):
        return {
            'embedding_dim': self.embedding_dim,
            'units': self.units,
            'batch_size': self.batch_size,
            'cpu_mode': self.cpu_mode
        }

    # Create input and target texts from the text
    @staticmethod
    def split_into_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]

        return input_text, target_text

    # Convert string to numbers
    def vocab_to_indices(self, sentence):
        if self.is_word_based():
            if type(sentence) == str:
                sentence = divide_word(sentence.lower())

            sentence = [sentence]
        else:
            sentence = sentence.lower()

        return np.array(self.tokenizer.texts_to_sequences(sentence)).reshape(-1,)

    # Preparing the model (both trainer/generator)
    def build_model(self, batch_size=None):
        # Hyper parameters
        if not batch_size:
            batch_size = self.batch_size

        # Disable CuDNN if GPU is not available
        if self.cpu_mode or not tf.test.is_gpu_available():
            gru = functools.partial(
                keras.layers.GRU,
                recurrent_activation='sigmoid',
            )
        else:
            gru = keras.layers.CuDNNGRU

        grus = []
        for _ in range(NUM_HIDDEN_LAYERS):
            grus.append(gru(
                self.units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform'
            ))
            grus.append(keras.layers.Dropout(0.5))

        return keras.Sequential(
            [keras.layers.Embedding(self.vocab_size, self.embedding_dim, batch_input_shape=[batch_size, None])]
            + grus
            + [keras.layers.Dense(self.vocab_size)]
        )

    # Training tasks
    def build_trainer(self):
        self.trainer = self.build_model()

    @staticmethod
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def compile(self):
        self.trainer.compile(optimizer=tf.train.AdamOptimizer(), loss=self.loss)

    @staticmethod
    def callbacks(save_dir):
        return [
            keras.callbacks.ModelCheckpoint(str(Path(save_dir).joinpath("ckpt_{epoch}")), save_weights_only=True, period=5, verbose=1),
            keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)
        ]

    def fit(self, save_dir, epochs):
        start_time = time.time()
        history = self.trainer.fit(self.dataset.repeat(), epochs=epochs, steps_per_epoch=self.steps_per_epoch, callbacks=self.callbacks(save_dir))
        elapsed_time = time.time() - start_time
        print("Time taken for learning {} epochs: {:.3f} minutes ({:.3f} minutes / epoch )".format(epochs, elapsed_time / 60, (elapsed_time / epochs) / 60))

        return history

    def save_trainer(self, save_dir):
        if Path.is_dir(save_dir) is not True:
            Path.mkdir(save_dir, parents=True)

        with save_dir.joinpath('parameters.json').open('w', encoding='utf-8') as params:
            params.write(json.dumps(self.parameters()))

        self.trainer.save_weights(str(Path(save_dir.joinpath('weights'))))
        self.save_tokenizer(save_dir)

    def load_trainer(self, load_dir):
        self.trainer.load_weights(self.path(Path(load_dir)))

    # Generating tasks
    def build_generator(self, load_dir):
        with Path(load_dir).joinpath('parameters.json').open() as parameters:
            self.set_parameters(**json.load(parameters))

        self.generator = self.build_model(batch_size=1)
        self.generator.load_weights(self.path(Path(load_dir)))

    def save_generator(self, save_dir):
        self.generator.save(str(Path(save_dir).joinpath('generator.h5')))

    def load_generator(self, load_dir):
        self.load_tokenizer(load_dir)
        self.generator = keras.models.load_model(str(Path(load_dir).joinpath('generator.h5')))

    def generate_text(self, start_string=None, gen_size=1, temperature=1.0, delimiter=None):
        if not start_string:
            start_string = choice(self.idx2vocab)

        generated_text = [start_string]
        # Vectorize start string
        try:
            input_eval = tf.expand_dims(self.vocab_to_indices(start_string), 0)
            print('Start string:', start_string)
        except KeyError:
            print('Unknown word included')
            return ''

        count = 0
        self.generator.reset_states()
        with tqdm(desc='Generating...', total=gen_size) as pbar:
            while count < gen_size:
                predictions = self.generator(input_eval)
                # remove the batch dimension
                predictions = tf.squeeze(predictions, 0)

                # Using the multinomial distribution to predict the word returned by the model
                # Temperature means randomness of text generation
                predictions = predictions / temperature
                predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

                # Pass the predicted word as the next input to the model along with the previous hidden state
                input_eval = tf.expand_dims([predicted_id], 0)

                try:
                    char = self.idx2vocab[predicted_id]
                except KeyError:
                    # Mark as an unknown word if predicted ID is out of bounds
                    char = '<oob>'
                generated_text.append(char)

                if char == delimiter or not delimiter:
                    count += 1
                    pbar.update()

        return generated_text

    # Return the path to <ckpt_dir>/checkpoint
    @staticmethod
    def path(ckpt_dir):
        return tf.train.latest_checkpoint(str(Path(ckpt_dir)))
