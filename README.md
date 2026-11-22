# MNIST CNN with PyTorch

PyTorchのテンソルと自動微分で手書き数字を分類するCNNです。畳み込み、ReLU、最大プーリング、全結合の処理を明示的に実装しています。

## 実行方法

リポジトリのルートで Python 3.10 以降を使います。

```bash
python -m pip install -r requirements.txt
python main.py
```

初回にMNISTをダウンロードします。学習には600枚、評価には100枚を使用し、損失と正解率を表示します。学習済みパラメータは `mnist_cnn_parameters.pt` に保存します。
