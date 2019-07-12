import lzma
import time

import tensorflow as tf
from tensorflow import keras

from modules.text_model import TextModel

SEQ_LENGTH = 100
BUFFER_SIZE = 10000


# Model for benchmarking
class BMModel(TextModel):
    def __init__(self):
        super().__init__()
        self.set_parameters()
        self.build_dataset()
        self.build_trainer()
        self.compile()

    def build_dataset(self):
        self.tokenizer = keras.preprocessing.text.Tokenizer(filters='\\\t', oov_token='<oov>', char_level=True)

        # Retrieve and decompress text
        path = keras.utils.get_file("souseki.txt.xz", "https://drive.google.com/uc?export=download&id=1RnvBPi0GSg07-FhiuHpkwZahGwl4sMb5")
        with lzma.open(path) as file:
            text = file.read().decode()

        # Vectorize the text
        self.tokenizer.fit_on_texts(text)
        # Index 0 is preserved in the Keras tokenizer for the unknown word, but it's not included in vocab2idx
        self.idx2vocab = {i: v for v, i in self.tokenizer.word_index.items()}
        self.vocab_size = len(self.idx2vocab) + 1
        text_size = len(text)
        print("Text has {} characters ({} unique characters)".format(text_size, self.vocab_size - 1))

        # Creating a mapping from unique characters to indices
        text_as_int = self.vocab_to_indices(text)

        # The maximum length sentence we want for single input in characters
        chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(SEQ_LENGTH + 1, drop_remainder=True)
        self.dataset = chunks.map(self.split_into_target).shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True)
        self.steps_per_epoch = text_size // SEQ_LENGTH // self.batch_size

    @staticmethod
    def callbacks(model_dir):
        return []

    def fit(self):
        if tf.test.is_gpu_available():
            epochs = 50
        else:
            epochs = 5

        start_time = time.time()
        history = self.trainer.fit(self.dataset.repeat(), epochs=epochs, steps_per_epoch=self.steps_per_epoch)
        elapsed_time = time.time() - start_time
        result_text = 'Time taken for learning {} epochs: {:.3f} minutes ({:.3f} minutes / epoch )\nLoss: {}'.format(
            epochs, elapsed_time / 60,
            (elapsed_time / epochs) / 60,
            history.history['loss'][-1]
        )

        print(result_text)

        return history, result_text
