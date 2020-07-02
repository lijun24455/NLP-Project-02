# 数据预处理流程 解压 -> 数据合并与标签提取 -> 切词 -> Tokenize -> Padding
import os
import pickle
import re
import zipfile
from collections import Counter

import jieba
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer


def unzip(ZIP_PATH, OUT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(os.path.join(OUT_PATH))
        print("已将压缩包解压至{}".format(OUT_PATH))


def combile_csvs(OUT_PATH):
    r = re.compile(r'\[知识点：\]\n(.*)')  # 用来寻找知识点的正则表达式
    r1 = re.compile(r'纠错复制收藏到空间加入选题篮查看答案解析|\n|知识点：|\s|\[题目\]')  # 简单清洗
    data = []
    for root, dirs, files in os.walk(OUT_PATH):
        if files:  # 如果文件夹下有csv文件
            # print('r:', root, 'd:', ' '.join(dirs), 'f:', ' '.join(files))
            for f in files:
                subject = re.findall('高中_(.{2})', root)[0]
                topic = f.strip('.csv')
                tmp = pd.read_csv(os.path.join(root, f))
                tmp['subject'] = subject  # 主标签：科目
                tmp['topic'] = topic  # 副标签：科目下主题
                tmp['knowledge'] = tmp['item'].apply(
                    lambda x: r.findall(x)[0].replace(',', ' ') if r.findall(x) else '')
                tmp['content'] = tmp['item'].apply(lambda x: r1.sub('', r.sub('', x)))
                data.append(tmp)
    data = pd.concat(data).reset_index(drop=True)
    # 删掉多余的两列
    data.drop(['web-scraper-order', 'web-scraper-start-url', 'item'], axis=1, inplace=True)
    return data


def extract_label_content(df, freq=0.001):
    print(df.shape)
    knowledges = ' '.join(df['knowledge']).split()
    knowledges = Counter(knowledges)
    print('总知识点数：', len(knowledges))
    k = int(df.shape[0] * freq)
    print('取 top_k =', k)
    df.knowledge = df.knowledge.apply(lambda x: ' '.join([label for label in x.split() if knowledges[label] > k]))
    print(df.shape)
    df['label'] = df[['subject', 'topic', 'knowledge']].apply(lambda x: ' '.join(x), axis=1)

    return df[['label', 'content']]


def load_stopwords(stop_words_file):
    return {line.strip() for line in open(stop_words_file, encoding='UTF-8').readlines()}


def sentence_to_words(sentence, stop_words):
    # 去标点
    r = re.compile("[^\u4e00-\u9fa5]+|题目")
    sentence = r.sub("", sentence)  # 删除所有非汉字字符

    # 切词
    words = jieba.cut(sentence, cut_all=False)

    # 去停用词
    words = [w for w in words if w not in stop_words]
    return words


def process_content(df, stop_words):
    df.content = df.content.apply(lambda x: sentence_to_words(x, stop_words))
    return df


def gen_cnn_data(df, X_NPY_PATH, Y_NPY_PATH, TOKENIZER_BINARIZER, max_len=200):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df.label.apply(lambda x: x.split()))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.content.tolist())
    x = tokenizer.texts_to_sequences(df.content)
    x = pad_sequences(x, max_len, padding='post', truncating='post')

    # 保存数据
    np.save(X_NPY_PATH, x)
    np.save(Y_NPY_PATH, y)
    print('已创建并保存x,y至：\n {} \n {}'.format(X_NPY_PATH, Y_NPY_PATH))
    tb = {'tokenizer': tokenizer, 'binarizer': mlb}  # 用个字典来保存
    with open(TOKENIZER_BINARIZER, 'wb') as f:
        pickle.dump(tb, f)
    print('已创建并保存tokenizer和binarizer至：\n {}'.format(TOKENIZER_BINARIZER))


if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    DATA_DIR = os.path.join(ROOT_PATH, 'data')

    ZIP_PATH = os.path.join(DATA_DIR, '百度题库.zip')
    OUT_PATH = os.path.join(DATA_DIR, '题库')
    STOP_WORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')

    # TextCNN生成文件
    X_NPY_PATH = os.path.join(DATA_DIR, 'CNN', 'x.npy')
    Y_NPY_PATH = os.path.join(DATA_DIR, 'CNN', 'y.npy')
    TOKENIZER_BINARIZER = os.path.join(DATA_DIR, 'CNN', 'tokenizer_binarizer')

    # 1 解压
    # unzip(ZIP_PATH, OUT_PATH)

    # 2.1 数据合并
    df = combile_csvs(OUT_PATH)  # df.shape:(29813,4)

    # 2.2 标签提取
    df = extract_label_content(df, freq=0)  # ['label', 'content' ]
    print(df.sample(3))

    # 3 切词
    stop_words = load_stopwords(STOP_WORDS_PATH)
    df = process_content(df, stop_words)
    print('after cut-->\n\r', df.head(2))

    # 4 Tokenize & Padding
    gen_cnn_data(df, X_NPY_PATH, Y_NPY_PATH, TOKENIZER_BINARIZER=TOKENIZER_BINARIZER)
