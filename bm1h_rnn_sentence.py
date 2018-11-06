import argparse
import numpy as np
import time
import tensorflow as tf
tf.enable_eager_execution()
import lzma
from tqdm import tqdm
from modules.model import Model

## Create input and target texts from the text
def split_into_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]

    return input_text, target_text

## Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

def main():
    parser = argparse.ArgumentParser(description="Benchmarking of sentence generation with RNN.")
    parser.add_argument("-c", "--cpu_mode", action='store_true', help="Force to use CPU (default: False)")
    args = parser.parse_args()

    with lzma.open('souseki_utf8.txt.xz') as file:
        text = file.read().decode()

    ## Vectorize the text
    text_size = len(text)
    # unique character in text
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    print("Text has {} characters ({} unique characters)".format(text_size, vocab_size))
    # Creating a mapping from unique characters to indices
    # This list doesn't have character that is not contained in the text
    char2idx = {char:index for index, char in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    # The maximum length sentence we want for single input in characters
    seq_length = 100
    chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length + 1, drop_remainder=True)

    batch_size = 64
    # Buffer size to shuffle the dataset
    buffer_size = 10000

    ## Creating batches and shuffling them
    dataset = chunks.map(split_into_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    ## Create the model
    embedding_dim = 256
    # embedding_dim = 16
    # RNN (Recursive Neural Network) nodes
    units = 1024
    # units = 64

    ## モデル作成
    model = Model(vocab_size, embedding_dim, units, force_cpu=args.cpu_mode)
    ## 最適化関数・損失関数を設定
    optimizer = tf.train.AdamOptimizer()

    epoch = 0
    elapsed_time = 0
    # 損失関数最小値
    min_loss = 100
    # 制限時間
    minutes = 60
    # minutes = 2
    start = time.time()
    # この時間を経過したら…ではなく、時間切れになった時のepochの学習を終えたら学習終了
    while elapsed_time < (60 * minutes):
        epoch += 1
        print("Epoch:", epoch)
        epoch_start = time.time()

        # initializing the hidden state at the start of every epoch
        hidden = model.reset_states()

        for (batch, (input, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # feeding the hidden state back into the model
                predictions, hidden = model(input, hidden)

                # reshape target to make loss function expect the target
                target = tf.reshape(target, (-1,))
                loss = loss_function(target, predictions)

                # 損失関数最小値を記録
                if min_loss > loss:
                    min_loss = loss

            gradients = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(gradients, model.variables))

            print("Batch: {}, Loss: {:.4f}".format(batch + 1, loss), end="\r")

        elapsed_time = time.time() - start
        print("Time taken for epoch {}: {:.3f} sec, Loss: {:.3f}\n".format(
            epoch,
            time.time() - epoch_start,
            loss.numpy()
        ))

    print("Time!")
    elapsed_time = elapsed_time / 60

    ## 訓練済みモデルで予測
    gen_size = 1000
    generated_text = ''
    start_string = "吾輩は"
    # start_stringを番号にする
    input_eval = tf.expand_dims([char2idx[s] for s in start_string], 0)
    # temperaturesが大きいほど予想？しにくい(surprising)テキストが生成される
    temperature = 1.0

    # 隠れ状態の形：(batch_size, units)
    hidden = [tf.zeros((1, units))]

    for i in tqdm(range(gen_size), desc="Generating..."):
        predictions, hidden = model(input_eval, hidden)

        # モデルから返ってきた単語を予測するのに偏微分を使う
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        # 予測された言葉と以前の隠れ状態をモデルに次の入力として渡す
        input_eval = tf.expand_dims([predicted_id], 0)

        generated_text += idx2char[predicted_id]

    generated_text = start_string + generated_text + "\n"
    print("Generated text:")
    print(generated_text)

    print("Learned {} epochs in {:.3f} minutes ({:.3f} epochs / minute)".format(epoch, elapsed_time, epoch / elapsed_time))
    print("Minimum loss:", min_loss.numpy())

if __name__ == '__main__':
    main()
