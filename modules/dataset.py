import tensorflow as tf
import numpy as np

class TextDataset():
    def __init__(self, text):
        ## Vectorize the text
        # unique character in text
        vocab = sorted(set(text))
        self.vocab_size = len(vocab)
        print("Text has {} characters ({} unique characters)".format(len(text), self.vocab_size))
        # Creating a mapping from unique characters to indices
        # This list doesn't have character that is not contained in the text
        self.char2idx = {char:index for index, char in enumerate(vocab)}
        self.idx2char = np.array(vocab)
        text_as_int = np.array(self.str_to_indices(text))

        # The maximum length sentence we want for single input in characters
        seq_length = 100
        chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length + 1, drop_remainder=True)

        batch_size = 64
        # Buffer size to shuffle the dataset
        buffer_size = 10000

        ## Creating batches and shuffling them
        dataset = chunks.map(self.split_into_target)
        self.dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    @staticmethod
    ## Create input and target texts from the text
    def split_into_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]

        return input_text, target_text

    ## Convert string to numbers
    def str_to_indices(self, str):
        return [self.char2idx[c] for c in str]

    ## Convert numbers to string
    def indices_to_str(self, indices):
        return [self.idx2char[id] for id in indices]
