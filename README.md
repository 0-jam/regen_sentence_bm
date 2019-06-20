# Benchmarking of Sentence Generation

- Benchmarking script based on [rnn_sentence.py][mmt]
- Records of benchmarking is [here](https://gist.github.com/0-jam/f21f44375cb70b987e99cda485d6940d)

---

1. [Environments](#Environments)
   1. [Software](#Software)
1. [Installation](#Installation)
1. [Usage](#Usage)
1. [Regulation (v3, 20190319)](#Regulation-v3-20190319)
   1. [About Dataset](#About-Dataset)
   1. [Rule](#Rule)
   1. [Evaluation](#Evaluation)

---

## Environments

### Software

- Python <= 3.7.3
- Tested OSs
    - Ubuntu 18.04.2 + ROCm 2.1
    - Ubuntu 18.04.2 + CUDA 10.0 + CuDNN 7.5.0.56
    - Arch Linux (Linux 5.1.11 + NVIDIA 430.26) + CUDA 10.1.168 + CuDNN 7.6.0.64
- TensorFlow >= 1.14.0rc1 (< 2.0)

## Installation

```bash
# If you use pyenv, install liblzma header before building Python
$ sudo apt install liblzma-dev
$ pyenv install 3.7.2
$ pip install tensorflow numpy tqdm matplotlob
```

## Usage

```bash
# Just execute:
$ python bm_rnn_sentence.py
```

## Regulation (v3, 20190319)

### About Dataset

- 7 novels written by Souseki Natsume（夏目漱石）
    1. 坊っちゃん (Bocchan)
    1. こころ (Kokoro)
    1. 草枕 (Kusamakura)
    1. 思い出す事など (Omoidasu koto nado)
    1. 三四郎 (Sanshiro)
    1. それから (Sorekara)
    1. 吾輩は猫である (Wagahai wa neko de aru)
- Based on [Aozora Bunko](https://www.aozora.gr.jp/index_pages/person148.html)
    - Already preprocessed by [this](#aozora-bunko) method
- Dataset is compressed to XZ
    - Extract automatically when execute benchmarking
    - About 3.01MiB after decompressing
    - Compress: `$ xz -9 -e -T 0 souseki_utf8.txt`
    - Extract: `$ xz -d souseki_utf8.txt -k`

### Rule

1. Time measurement begins when training of the model is started
1. Train _50_ (default) epochs
    - If GPU is not available, the number of epoch reduces 50 to _3_
1. Print results
    - Elapsed time
    - Epochs per minute
    - The value of loss function
    - Generated text
        - The number of characters: 1000

### Evaluation

- How many times per epoch?
- What loss function's value?
    - The smaller loss function's value, the more _readable_ sentence can be generated ...probably

[mmt]: https://github.com/0-jam/regen_my_sentences
