import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
pd.set_option('display.max_colwidth', None)  # set the max width of column to display

# nltk.download('stopwords') # download stopwords package to remove stopwords
# nltk.download('punkt')  # download punkt package to tokenize text


def clean_tweet(tweet):
    """
    clean data
    :param tweet: tweet string
    :return: cleaned tweet
    """
    # remove URL
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # replacing HTML entities
    tweet = re.sub(r'&\w+;', ' ', tweet)
    # remove @username
    tweet = re.sub(r'@\w+', '', tweet)
    # remove HTML entity codes (like &#57361;)
    tweet = re.sub(r'&#[0-9]+;', ' ', tweet)
    # remove special characters and punctuation marks
    tweet = re.sub(r'\W', ' ', tweet)
    # remove extra white space
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet


def lowercase_tokenize(tweet):
    """
    将推文文本转换为小写，进行分词（Tokenization）
    transform tweet to lowercase and tokenize
    :param tweet: tweet string
    :return: tokenized tweet
    """
    tweet = tweet.lower()
    tweet_tokens = word_tokenize(tweet)
    return tweet_tokens


def remove_stopwords(tokens):
    """
    去除停顿词
    remove stop words
    :param tokens: tokenized tweet
    :return: tokenized tweet without stop words
    """
    stop_words = set(stopwords.words('english'))  # use English stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]

    return filtered_tokens


def label_tweet(row):
    """
    label tweet
    :param row: row of dataframe
    :return: label of tweet
    """
    if row['class'] == 0:
        return 'hate_speech'
    if row['class'] == 1:
        return 'offensive_language'
    if row['class'] == 2:
        return 'neither'


if __name__ == '__main__':
    # load data
    data_path = r'./data/tweet.csv'
    data = pd.read_csv(data_path)
    data = data.dropna()

    # clean data
    data['cleaned_tweet'] = data['tweet'].apply(clean_tweet)

    # tokenize data
    data['tokens'] = data['cleaned_tweet'].apply(lowercase_tokenize)

    # remove stop words
    data['filtered_tokens'] = data['tokens'].apply(remove_stopwords)
    data['filtered_tweet'] = data['filtered_tokens'].apply(lambda x: ' '.join(x))

    # label data
    data['label'] = data.apply(label_tweet, axis=1)
    data.dropna()

    data.to_csv('./data/cleaned_tweet.csv', index=False)

