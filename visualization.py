import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

cleaned_tweet = pd.read_csv('data/cleaned_tweet.csv')

# 标签分布可视化 Label Distribution Visualization
plt.figure(figsize=(8, 6), facecolor=None)
sns.countplot(data=cleaned_tweet, x='label')
plt.title('Label Distribution')
plt.savefig(r'./img/label_distribution.png')
plt.show()

# 合并所有文本
all_text = ' '.join(cleaned_tweet['cleaned_tweet'])
# 生成词云 wordcloud
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10).generate(all_text)

# 展示词云
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(r'./img/wordcloud.png')
plt.show()

# 创建一个点图来展示数据集中每个推文的标注数量
plt.figure(figsize=(10, 8))

# 绘制 hate_speech_count
sns.scatterplot(data=cleaned_tweet, x=cleaned_tweet.index, y='hate_speech_count', color='red', label='Hate Speech')

# 绘制 offensive_language_count
sns.scatterplot(data=cleaned_tweet, x=cleaned_tweet.index, y='offensive_language_count', color='blue', label='Offensive Language')

# 绘制 neither_count
sns.scatterplot(data=cleaned_tweet, x=cleaned_tweet.index, y='neither_count', color='green', label='Neither')

# 给图表添加标题和标签
plt.title('Counts of Tweet Annotations')
plt.xlabel('Tweet Index')
plt.ylabel('Count')
# 显示图例
plt.legend()
plt.savefig(r'./img/counts_of_tweet_annotations.png')
# 显示图形
plt.show()

# 创建一个箱线图来展示每个标注的数量
"""
    箱形图 (Boxplot)
    箱形图是一种统计图形，用于展示一组数据的分布情况。以下是各个组成部分的含义：
    
    盒子：盒子的上边缘和下边缘分别表示数据的第三四分位数 (Q3) 和第一四分位数 (Q1)。因此，盒子的长度表示了中间 50% 数据的分布范围（也就是四分位距）。
    中间线：盒子中间的线表示数据的中位数（Q2）。
    须线：从盒子外伸出的线条被称为“须”（Whiskers），通常表示数据的范围，不包括异常值。须线的确切计算方式可能有所不同，但通常它们从第一四分位数到最小值，和从第三四分位数到最大值。
    点：箱形图之外的点通常表示异常值，这些值远离其余的数据点。
    在您的箱形图中，每个分类（hate_speech_count、offensive_language_count、neither_count）都有一个盒子，表示各自的数据分布。例如，如果offensive_language_count的盒子比其他两个宽，则表示该分类下的数据在中间 50% 范围内变化更大。
"""
plt.figure(figsize=(10, 8))
sns.boxplot(data=cleaned_tweet[['hate_speech_count', 'offensive_language_count', 'neither_count']])
plt.title('Boxplot of Annotations Count')
plt.ylabel('Count')
plt.savefig(r'./img/boxplot_of_annotations_count.png')
plt.show()

# 创建一个箱线图来展示每个标注的数量
"""
    小提琴图 (Violin Plot)
    小提琴图类似于箱形图，但它还包括了数据的核密度估计，更详细地展示了数据的分布情况，特别是数据的密度：
    
    宽度：小提琴图的宽度在不同高度上的变化反映了数据在这些值上的密度——图形越宽，表明该值的数据点越多。
    内部的白点：通常表示数据的中位数。
    内部的棒形：表示四分位范围，类似于箱形图的盒子。
    整体形状：小提琴的形状可以展示数据分布的模式，例如是否对称、是否有单峰或多峰等。
    在您的小提琴图中，每个小提琴表示一个分类，从图形的形状可以看出数据的分布情况。例如，如果一个小提琴的上半部分比下半部分宽，这表明在较高的计数值上有更多的数据点。
"""
plt.figure(figsize=(10, 8))
sns.violinplot(data=cleaned_tweet[['hate_speech_count', 'offensive_language_count', 'neither_count']])
plt.title('Violin Plot of Annotations Count')
plt.ylabel('Count')
plt.savefig(r'./img/violin_plot_of_annotations_count.png')
plt.show()


