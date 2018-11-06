# (Benchmarking) Regenerate Sentences

- [以前書いた文書生成プログラム][mmt]ベースのベンチマークスクリプト
    - 基本的にオリジナルの機能を大きく省いただけ
- 1時間（仮）で何epoch学習できて、そのモデルから1000文字（仮）生成した時にどの程度読める文章ができるかを比較

---

1. [Environments](#environments)
1. [About Dataset](#about-dataset)
1. [Installation](#installation)
1. [Usage](#usage)
1. [Rule](#rule)
1. [Evaluation](#evaluation)
1. [Records](#records)

---

## Environments

- Python = 3.6.7 on Ubuntu 18.04.1 on Windows Subsystem for Linux (Windows 10 Home 1803 (April 2018))
- Python = 3.6.7 on Windows 10 Home 1803 (April 2018)
- TensorFlow >= 1.11.0

## About Dataset

- 夏目漱石の小説7編
    1. 坊っちゃん
    1. こころ
    1. 草枕
    1. 思い出す事など
    1. 三四郎
    1. それから
    1. 吾輩は猫である
- [青空文庫](https://www.aozora.gr.jp/index_pages/person148.html)のテキスト版をベースに前処理
    - 7編を1つのテキストファイルに結合
    - ふりがなや装飾文字の除去
        - 詳しくは[こちら](https://github.com/0-jam/regen_my_sentences#aozora-bunko)
    - 文字コード変換(Shift_JIS (CP932) -> UTF-8)
- 前処理後にXZ (LZMA2)で圧縮
    - スクリプト実行時にPythonによって展開される
    - 展開後サイズ：約3.01MiB
    - 圧縮：`$ xz -9 -e -T 0 souseki_utf8.txt`
    - 展開：`$ xz -d souseki_utf8.txt -k`

## Installation

```bash
# Pythonインストール前にLZMAライブラリのヘッダーをインストールする必要がある（Windowsでは不要）
$ sudo apt install liblzma-dev
$ pyenv install 3.6.7
$ pip install tensorflow numpy tqdm
```

## Usage

```bash
# これだけ
# "-c"オプションをつけると強制的にCPUを使った学習になる
$ python bm1h_rnn_sentence.py
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

- ベンチマーク記録は[こちら](https://gist.github.com/0-jam/f21f44375cb70b987e99cda485d6940d)

[mmt]: https://github.com/0-jam/regen_my_sentences
