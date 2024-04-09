import torch
import torch.nn as nn
from TextCNN import TextCNN
from collections import Counter
import pandas as pd


# 加载词汇表
# Load vocabulary
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r') as file:
        for line in file:
            word, index = line.strip().split(',')
            vocab[word] = int(index)
    return vocab


# 文本预处理
# Text preprocessing
def preprocess_text(text, vocab, max_length=100):
    tokens = text.split()
    encoded_text = [vocab.get(tokens, vocab["<UNK>"]) for token in tokens]
    pad_length = max_length - len(encoded_text)
    if pad_length > 0:
        encoded_text += [vocab["<PAD>"]] * pad_length
    return torch.tensor([encoded_text], dtype=torch.long)


# 加载模型
# Load model
def load_model(model_path, num_classes, vocab_size, embedding_dim):
    device = torch.device("cpu")
    model = TextCNN(num_classes=num_classes, num_embeddings=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# 推理
# Inference
def infer(text, model, vocab):
    input_tensor = preprocess_text(text, vocab)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


# 主程序
# Main program
def main(text):
    vocab_path = 'data/vocab.csv'
    model_path = 'model/best_model.pth'
    vocab = load_vocab(vocab_path)
    model = load_model(model_path, num_classes=3, vocab_size=len(vocab), embedding_dim=128)
    # 执行推理
    # Perform inference
    prediction = infer(text, model, vocab)

    # 类别对照表
    # Class dictionary
    class_dict = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
    if prediction in class_dict:
        prediction = class_dict[prediction]

    print("Predicted class:", prediction)


if __name__ == "__main__":
    text = input("Please input :")
    main(text)
