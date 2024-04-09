from collections import Counter
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


class TextCNN(nn.Module):
    def __init__(self, num_classes, num_embeddings=-1, embedding_dim=128, kernel_sizes=[3, 4, 5, 6],
                 num_channels=[256, 256, 256, 256], embeddings_pretrained=None):
        """
        :param num_classes: 输出维度(类别数num_classes)
        :param num_embeddings: 词典的大小(vocab_size) size of the dictionary of embeddings
                               当num_embeddings<0,模型会去除embedding层
        :param embedding_dim: 词向量特征长度 the size of each embedding vector
        :param kernel_sizes: CNN层卷积核大小 The Size of Convolutional Kernel
        :param num_channels: CNN层卷积核通道数 The Number of Channels of Convolutional Kernel
        :param embeddings_pretrained: embeddings pretrained参数，默认None
        :return:
        """
        super(TextCNN, self).__init__()
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        # embedding层
        if self.num_embeddings > 0:
            # embedding之后的shape: torch.Size([200, 8, 300])
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            if embeddings_pretrained is not None:
                self.embedding = self.embedding.from_pretrained(embeddings_pretrained, freeze=False)
        # 卷积层 Conv1d
        self.cnn_layers = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=c,
                          kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        # 最大池化层
        self.pool = GlobalMaxPool1d()
        # 输出层
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sum(num_channels), self.num_classes)
        )

    def forward(self, input):
        """
        :param input:  (batch_size, context_size, embedding_size(in_channels))
        :return:
        """
        if self.num_embeddings > 0:
            # 得到词嵌入(b,context_size)-->(b,context_size,embedding_dim)
            input = self.embedding(input)
            # (batch_size, context_size, channel)->(batch_size, channel, context_size)
        input = input.permute(0, 2, 1)
        y = []
        for layer in self.cnn_layers:
            x = layer(input)
            x = self.pool(x).squeeze(-1)
            y.append(x)
        y = torch.cat(y, dim=1)
        out = self.classify(y)
        return out


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = [self.vocab.get(token, self.vocab["<UNK>"]) for token in text.split(" ")]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        pad_length = self.max_length - len(tokens)
        if pad_length > 0:
            tokens_tensor = F.pad(tokens_tensor, (0, pad_length), 'constant', 0)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        return tokens_tensor, label_tensor
