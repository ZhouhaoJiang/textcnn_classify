import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from TextCNN import TextDataset, TextCNN

cleaned_tweet = pd.read_csv('data/cleaned_tweet.csv')

# 假设 corpus 是包含所有文本数据的列表
cleaned_tweet = cleaned_tweet.dropna(subset=['filtered_tweet'])
corpus = cleaned_tweet['filtered_tweet'].tolist()

# 统计每个词的出现频率
word_counts = Counter(word for sentence in corpus for word in sentence.split())
# print(word_counts.most_common(10))

vocab_size = 10000
vocab = {word: index + 2 for index, (word, _) in enumerate(word_counts.most_common(vocab_size))}
# <PAD>（填充标记），用于将所有文本序列填充到相同长度。
# <UNK>（未知标记），用于表示那些不在词汇表中的词。
vocab["<PAD>"] = 0  # 特殊填充标记
vocab["<UNK>"] = 1  # 特殊未知词标记

# 将词汇表写入 CSV 文件
# write vocabulary to csv file
# vocab_path = 'data/vocab.csv'
# pd.Series(vocab).to_csv(vocab_path, header=False)

X_train, X_test, y_train, y_test = train_test_split(cleaned_tweet['filtered_tweet'], cleaned_tweet['class'],
                                                    test_size=0.2, random_state=42)

# 创建 Dataset
train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), vocab)
test_dataset = TextDataset(X_test.tolist(), y_test.tolist(), vocab)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 创建模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(num_classes=3, num_embeddings=len(vocab), embedding_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_accuracy = 0
num_epochs = 10
# 初始化TensorBoard的SummaryWriter
# initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir='logs/textcnn_experiment')
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_loss = 0
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
    for texts, labels in progress_bar:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()  # 梯度清零
        outputs = model(texts)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()  # 累加损失
        progress_bar.set_postfix(loss=total_loss / len(train_loader))

    # 记录训练损失
    # record training loss
    writer.add_scalar('Training loss', total_loss / len(train_loader), epoch)

    # 验证步骤
    model.eval()
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
    with torch.no_grad():
        for texts, labels in progress_bar:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix(accuracy=total_correct / total_samples)

    val_accuracy = total_correct / total_samples
    # 记录验证准确率
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), f'model/best_model_{epoch}.pth')

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'model/model_{epoch}.pth')
    print(f"Validation accuracy: {val_accuracy:.4f}")


"""
在文本分类任务中，选择最合适的模型通常取决于多个因素，包括数据集的特性、问题的复杂性以及训练数据的规模。
下面是一些决定使用TextCNN或随机森林分类器的考虑因素：

随机森林分类器：
    优点：
    训练速度快，对于中等大小的数据集可以很快得到结果。
    不太需要数据预处理，例如归一化。
    对于非线性问题表现良好。
    不容易过拟合。
    缺点：
    对于文本数据，需要手动特征工程（如TF-IDF）来转换文本为数值型特征。
    可能不如深度学习模型在捕捉语言的复杂性方面表现良好。
    
TextCNN：
    优点：
    能够自动从数据中学习表示，减少了特征工程的需要。
    对于文本数据，尤其是在有足够数据时，可以捕捉局部特征（如n-gram）。
    在大型复杂数据集上通常比传统机器学习方法表现更好。
    缺点：
    需要较大的数据集来学习有效的特征表示。
    训练时间可能较长，特别是在大型数据集上。
    对于小数据集，可能会过拟合。
    由于您实际测试的结果显示TextCNN比随机森林分类器效果好，这可能是因为TextCNN能更好地捕捉文本数据中的局部相关性和模式，而这些可能是随机森林在没有复杂特征工程的情况下难以捕捉的。深度学习模型（如TextCNN）通过学习嵌入层和隐藏层的表示来自动提取和组合特征，这在处理文本时非常有用，因为它可以捕捉到词汇使用的复杂模式和关联。

最终，选择哪个模型取决于实验的具体目标、可用资源（如计算能力和时间）、数据的性质以及实验的结果。
在实践中，通常建议尝试多种模型并比较它们的性能，从而根据具体任务选择最佳模型。在撰写论文时，您可以详细说明为什么TextCNN在您的特定任务中比随机森林表现得更好，包括提供实验结果和可能的理论解释。
"""