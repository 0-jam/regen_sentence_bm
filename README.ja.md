# 文章生成ベンチマーク

- [rnn_sentence.py][mmt]をベースとしたベンチマークスクリプト
- 過去の記録は[こちら](https://gist.github.com/0-jam/f21f44375cb70b987e99cda485d6940d)

---

1. [環境](#環境)
   1. [ソフトウェア](#ソフトウェア)
1. [インストール](#インストール)
1. [使用法](#使用法)
1. [レギュレーション (v3, 20190319)](#レギュレーション-v3-20190319)
   1. [データセット](#データセット)
   1. [ルール](#ルール)
   1. [評価基準](#評価基準)

---

## 環境

### ソフトウェア

- Python <= 3.7.2
- Tested OSs
    - Ubuntu 18.04.2 + ROCm 2.1
    - Ubuntu 18.04.2 + CUDA 10.0 + CuDNN 7.5.0.56
- TensorFlow >= 1.13.0 (< 2.0)

## インストール

```bash
# pyenv環境ではPythonビルド前にLZMAライブラリのヘッダーをインストールする必要がある
$ sudo apt install liblzma-dev
$ pyenv install 3.7.2
$ pip install tensorflow numpy tqdm matplotlob
```

## 使用法

```bash
# これだけ
$ python bm_rnn_sentence.py
```

## レギュレーション (v3, 20190319)

### データセット

- 夏目漱石の小説7編
    1. 坊っちゃん
    1. こころ
    1. 草枕
    1. 思い出す事など
    1. 三四郎
    1. それから
    1. 吾輩は猫である
- [この方法](#青空文庫)で前処理済
    - [青空文庫](https://www.aozora.gr.jp/index_pages/person148.html)がベース
- 前処理後にXZで圧縮
    - スクリプト実行時にPythonによって展開される
    - 展開後サイズ：約3.01MiB
    - 圧縮：`$ xz -9 -e -T 0 souseki_utf8.txt`
    - 展開：`$ xz -d souseki_utf8.txt -k`

### ルール

1. モデルの訓練が開始した瞬間に時間計測開始
1. _50_ （デフォルト） epoch学習する
    - GPUが使えない場合、epoch数は50から _3_ になる
1. 結果を表示
    - 経過した時間
    - 1分あたりのepoch数
    - 損失関数の値
    - 生成されたテキスト
        - 1000字生成される

### 評価基準

- 1epochあたりにかかった時間は？
- loss（損失関数）の値は？
    - 小さければ小さいほど _まともな_ 文章が生成される…はず

[mmt]: https://github.com/0-jam/regen_my_sentences
