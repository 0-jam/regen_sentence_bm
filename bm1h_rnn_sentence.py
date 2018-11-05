import argparse
import numpy as np
import random
import re
import time
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
import sys
import lzma
from tqdm import tqdm

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size, force_cpu=False):
        super(Model, self).__init__()
        self.units = units
        self.batch_sz = batch_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # CUDAが使えるか確認
        if tf.test.is_gpu_available() and not force_cpu:
            self.gru = tf.keras.layers.CuDNNGRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        else:
            self.gru = tf.keras.layers.GRU(
                self.units,
                return_sequences=True,
                return_state=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform'
            )

        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)

        # statesにモデルの状態を格納
        # 訓練中に毎回渡される
        output, states = self.gru(x, initial_state=hidden)

        # Densely-connected層に渡せる形にデータを整形
        # 整形後：(batch_size * max_length, hidden)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape after the dense layer is (batch_size * max_length, vocab_size)
        x = self.fc(output)

        return x, states

def main():
    parser = argparse.ArgumentParser(description="Generate sentence with RNN. README.md contains further information.")
    parser.add_argument("-o", "--output", type=str, help="output file path (default: stdout)")
    parser.add_argument("-c", "--cpu_mode", action='store_true', help="Force to use CPU (default: False)")
    args = parser.parse_args()

    with lzma.open('souseki_utf8.txt.xz') as file:
        text = file.read().decode()

    # テキスト中に現れる文字を取得
    unique = sorted(set(text))
    # 各文字に数字を割り当てる
    # テキストにない文字は記録されないので，開始文字列に未知の文字を与えるとエラー
    char2idx = {char:index for index, char in enumerate(unique)}
    idx2char = {index:char for index, char in enumerate(unique)}

    # 文の長さを指定
    max_length = 100
    vocab_size = len(unique)
    embedding_dim = 256
    units = 1024
    # embedding_dim = 4
    # units = 16
    batch_size = 64
    # シャッフル用バッファサイズ
    buffer_size = 10000

    ## 入力・出力用テンソル作成
    input_text = []
    target_text = []

    for i in range(0, len(text) - max_length, max_length):
        input = text[i:i + max_length]
        target = text[i + 1:i + 1 + max_length]

        input_text.append([char2idx[j] for j in input])
        target_text.append([char2idx[k] for k in target])

    ## バッチを作ってシャッフルする
    dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    ## モデル作成
    model = Model(vocab_size, embedding_dim, units, batch_size, force_cpu=args.cpu_mode)
    ## 最適化関数・損失関数を設定
    optimizer = tf.train.AdamOptimizer()

    def loss_function(real, preds):
        # one-hotベクトルを生成しなくていいようにsparse_softmax_cross_entropyを使う
        return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

    epoch = 0
    elapsed_time = 0
    minutes = 60
    # minutes = 3
    start = time.time()
    # 時間切れになったら…ではなく、時間切れになった時のepochの学習を終えたら学習終了
    while elapsed_time < (60 * minutes):
        epoch += 1
        epoch_start = time.time()
        min_loss = 100
        # epochごとに隠れ状態(hidden state)を初期化
        hidden = model.reset_states()

        for (batch, (input, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                # モデルに隠れ状態を与える
                predictions, hidden = model(input, hidden)

                # 損失関数に渡せる形に対象を整形する
                # (reshape target to make loss function expect the target)
                target = tf.reshape(target, (-1,))
                loss = loss_function(target, predictions)

                if min_loss > loss:
                    min_loss = loss

            gradients = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(gradients, model.variables), global_step = tf.train.get_or_create_global_step())

            print("Epoch: {}, Batch: {}, Loss: {:.4f}".format(epoch, batch + 1, loss))

        elapsed_time = time.time() - start
        print("Time taken for epoch {}: {:.3f} sec \n".format(epoch, time.time() - epoch_start))

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

    print("Start generating...")
    for i in tqdm(range(gen_size)):
        # print("prediction {} / {}".format(i + 1, gen_size))
        predictions, hidden = model(input_eval, hidden)

        # モデルから返ってきた単語を予測するのに偏微分を使う
        predictions = predictions / temperature
        predicted_id = tf.multinomial(tf.exp(predictions), num_samples = 1)[0][0].numpy()

        # 予測された言葉と以前の隠れ状態をモデルに次の入力として渡す
        input_eval = tf.expand_dims([predicted_id], 0)

        generated_text += idx2char[predicted_id]

    generated_text = start_string + generated_text + "\n"
    if args.output:
        with Path(args.output).open('w') as out:
            out.write(generated_text)
    else:
        sys.stdout.write(generated_text)

    print("Learned {} epochs in {} minutes ({:.3f} epochs / minute)".format(epoch, elapsed_time, epoch / elapsed_time))
    print("Minimum loss:", min_loss)

if __name__ == '__main__':
    main()
