from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

cleaned_tweet = pd.read_csv('data/cleaned_tweet.csv')

# vectorize data erm Frequency-Inverse Document Frequency (TF-IDF)
# 使用TF-IDF向量化数据
vectorizer = TfidfVectorizer(max_features=3000)
tfidf_matrix = vectorizer.fit_transform(cleaned_tweet['filtered_tweet']).toarray()

# get labels from cleaned_tweet
# 从cleaned_tweet中获取标签
labels = cleaned_tweet['label']

# split data into training and testing sets
# tfidf_matrix是一个稀疏矩阵，可以通过toarray()方法转换为numpy数组 已经向量化过的tweet数据
# random_state是随机数种子
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

# train a random forest classifier
# 训练一个随机森林分类器   n_estimators是森林中树的数量，会影响模型的准确率 random_state是随机数种子
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# fit the classifier to the training data
# 将分类器拟合到训练数据上
clf.fit(X_train, y_train)

# predict on test data
y_pred = clf.predict(X_test)

# print classification report
# 打印分类报告
print(classification_report(y_test, y_pred))
# joblib.dump(clf, 'random_forest_classifier.joblib')

"""
                    precision    recall  f1-score   support

       hate_speech       0.42      0.16      0.23       290
           neither       0.81      0.91      0.85       835
offensive_language       0.93      0.95      0.94      3832

          accuracy                           0.90      4957
         macro avg       0.72      0.67      0.67      4957
      weighted avg       0.88      0.90      0.88      4957
      
Precision（精确率）：对于每个类别，它表示的是被正确预测为该类别的样本数与被预测为该类别的样本总数的比例。例如，hate_speech 的精确率是 0.42，意味着当模型预测一个样本为 hate_speech 时，有 42% 的概率是正确的。

Recall（召回率）：也被称为真正率，它表示的是被正确预测为该类别的样本数与实际为该类别的样本总数的比例。例如，hate_speech 的召回率是 0.16，意味着在所有真实为 hate_speech 的样本中，有 16% 被模型正确预测。

F1-Score（F1 分数）：精确率和召回率的调和平均，是一个综合考量两者的性能指标。F1 分数的范围从 0 到 1，越接近 1 表示模型的性能越好。例如，hate_speech 的 F1 分数是 0.23，相对较低，意味着模型在这一类别上的表现不是很好。

Support（支持度）：表示每个类别在测试集中的实际样本数。例如，hate_speech 类别有 290 个样本。

Accuracy（准确率）：整个测试集中，被正确预测的样本数占总样本数的比例，这里是 90%。

Macro avg：宏平均，即计算各个类别指标的平均值，不考虑每个类别的样本数。

Weighted avg：加权平均，即计算各个类别指标的平均值，但是每个类别的指标会根据其支持度（即样本数）进行加权。
========================================================================================================================
随机森林分类器是一个非常流行且功能强大的机器学习模型，常用于分类和回归任务。以下是随机森林分类器的一些关键特点和工作原理：

集成学习方法：
随机森林是一个集成学习算法，它结合了多个决策树的预测结果来提高整体的预测准确性和稳定性。每个决策树都是一个单独的学习器，通过训练得到独立的预测。

构建多个决策树：
在随机森林中，构建多个决策树。每个树都是从原始数据集中随机抽取的样本（使用有放回抽样，即bootstrap抽样）上训练的。

特征的随机选择：
在构建树的过程中，不是所有特征都用于每次分裂决策。而是随机选择一部分特征。这种随机性有助于使模型更加健壮，减少过拟合。

投票机制：
对于分类任务，随机森林的预测结果是基于所有决策树的预测结果的“多数投票”。即，模型输出是由大多数树同意的类别。

重要参数：
n_estimators：这是森林中树的数量。树越多，模型通常越稳定，但计算成本也更高。
random_state：这是随机数生成器的种子。设置这个参数可以确保每次运行模型时结果的一致性。

优点：
随机森林通常具有很高的准确率，对异常值和噪声有很好的容忍度，并且不太容易过拟合。
能够处理高维数据，并提供关于特征重要性的指标。

缺点：
相对于单个决策树，随机森林需要更多的计算资源和时间。
随机森林模型可以变得非常庞大，可能不易解释。
随机森林分类器在很多实际应用中表现出色，特别是在处理具有多个特征的复杂数据集时。
"""

