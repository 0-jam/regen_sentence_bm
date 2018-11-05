# (Benchmarking) Regenerate Sentences

- [以前書いた文書生成プログラム][mmt]ベースのベンチマークスクリプト
- 1時間（仮）で何epoch学習できて、そのモデルから1000文字（仮）生成した時にどの程度読める文章ができるかを比較

---

1. [Environments](#environments)
1. [Installation](#installation)
1. [Usage](#usage)
1. [Rule](#rule)
1. [Evaluation](#evaluation)
1. [Records](#records)
    1. [PC 1 (CPU, Native Windows)](#pc-1-cpu-native-windows)
    1. [PC 1 (CPU, WSL)](#pc-1-cpu-wsl)
    1. [PC 2 (CPU)](#pc-2-cpu)
    1. [PC 2 (GPU)](#pc-2-gpu)

---

## Environments

- Python 3.6.7 on Ubuntu 18.04.1 on Windows Subsystem for Linux (Windows 10 Home 1803 (April 2018))
- Python 3.6.7 on Windows 10 Home 1803 (April 2018)

## Installation

```bash
# Pythonインストール前にLZMAライブラリのヘッダーをインストールする必要がある（Windowsでは不要）
$ sudo apt install liblzma-dev
$ pyenv install 3.6.7
$ pip install tensorflow numpy tqdm
```

## Usage

- 基本的に[オリジナル][mmt]の機能を大きく省いただけ
    - モデルの保存や生成されたテキストのファイルへの書き込み、各種パラメータ指定を削除

```bash
# 基本はこれだけ
$ python bm1h_rnn_sentence.py
# こうすると強制的にCPUを使った学習になる
$ python bm1h_rnn_sentence.py -c
```

## Rule

1. モデルの学習が始まった瞬間から計測スタート
1. あらかじめ決められた時間の間学習を続ける
1. 決められた時間を超過した場合、そのepochを終えた段階で学習を終了する
    - 例：制限時間15分
        - epoch数3の学習中に15分経過した場合、epoch3の学習を終えてループ終了
        - つまり、実際の実行時間は制限時間を超える
1. 結果を表示
    - 所要時間
    - epoch数
    - 上二つから計算できる1分あたりのepoch数
    - 損失関数の最小値

## Evaluation

- 何epoch学習できたか？
    - 1epochあたりにかかった時間は？
- 最小のloss（損失関数）の値は？
    - 小さければ小さいほど _まともな_ 文章が生成される…はず
    - 最終的に出た値のほうがいいかもしれないが、ここでは最低値を評価する

## Records

### PC 1 (CPU, Native Windows)

- CPU: Intel [Core i5 7200U](https://ark.intel.com/products/95443/Intel-Core-i5-7200U-Processor-3M-Cache-up-to-3_10-GHz)
- RAM: 8GB
- Windows 10 Home 1803 (April 2018)

### PC 1 (CPU, WSL)

- CPU, RAM: 同上
- OS: Ubuntu 18.04.1 on Windows Subsystem for Linux (Windows 10 Home 1803 (April 2018))

### PC 2 (CPU)

- CPU: AMD [Ryzen 7 1700](https://www.amd.com/ja/products/cpu/amd-ryzen-7-1700)
- RAM: 16GB
- OS: Ubuntu 18.04.1＋[ROCm](https://github.com/RadeonOpenCompute/ROCm)モジュール
    - [公式Dockerイメージ](https://hub.docker.com/r/rocm/tensorflow/)上で実行

### PC 2 (GPU)

- CPU, RAM: 同上
- GPU: AMD [Radeon RX 580](https://www.amd.com/ja/products/graphics/radeon-rx-580)
- VRAM: 8GB
- OS: 同上

[mmt]: https://github.com/0-jam/regen_my_sentences
