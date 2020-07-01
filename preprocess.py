# 数据预处理流程 解压 -> 数据合并与标签提取 -> 切词 -> Tokenize -> Padding
import os
import re
import zipfile
from collections import Counter

import jieba
import pandas as pd


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


def extract_label_content(df, freq = 0.01):
    print(df.shape)
    knowledges = ' '.join(df['knowledge']).split()
    knowledges = Counter(knowledges)
    k = int(df.shape[0] * freq)
    print('top_k =', k)
    top_k_knowledges = knowledges.most_common(df.shape[0] - k)
    df.knowledge = df.knowledge.apply(lambda x: ' '.join([label for label in x.split() if label in top_k_knowledges]))
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
    pass


if __name__ == '__main__':
    ROOT_PATH = os.getcwd()
    DATA_DIR = os.path.join(ROOT_PATH, 'data')

    ZIP_PATH = os.path.join(DATA_DIR, '百度题库.zip')
    OUT_PATH = os.path.join(DATA_DIR, '题库')
    STOP_WORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')

    # 1 解压
    # unzip(ZIP_PATH, OUT_PATH)

    # 2.1 数据合并
    df = combile_csvs(OUT_PATH) # df.shape:(29813,4)

    # 2.2 标签提取
    df = extract_label_content(df)
    # print(df.sample(3))

    # 切词
    stop_words = load_stopwords()
    df = process_content(df, stop_words)