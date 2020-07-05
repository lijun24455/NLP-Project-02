import argparse
import os
import pickle

from keras_preprocessing.text import Tokenizer


def load_tokenizer_binarizer(TOKENIZER_BINARIZER):
    """
    读取tokenizer 和 binarizer
    :return:
    """
    with open(TOKENIZER_BINARIZER, 'rb') as f:
        tb = pickle.load(f)
    return tb['tokenizer'], tb['binarizer']

class config:
    ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(ROOT_PATH, 'data')

    ZIP_PATH = os.path.join(DATA_DIR, '百度题库.zip')
    OUT_PATH = os.path.join(DATA_DIR, '题库')
    STOP_WORDS_PATH = os.path.join(DATA_DIR, 'stopwords.txt')

    # TextCNN生成文件
    X_NPY_PATH = os.path.join(DATA_DIR, 'CNN', 'x.npy')
    Y_NPY_PATH = os.path.join(DATA_DIR, 'CNN', 'y.npy')
    TOKENIZER_BINARIZER = os.path.join(DATA_DIR, 'CNN', 'tokenizer_binarizer')
    CKPT_POINT_DIR = os.path.join(DATA_DIR, 'CNN', 'train')

    def getArgs(self):
        parser = argparse.ArgumentParser(__doc__)

        parser.add_argument("--max_len", type=int, default=128)
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--filters", type=int, default=2)
        parser.add_argument("--kernel_sizes", type=str, default='2,3,4')
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        parser.add_argument("--vocab_size", type=int, default=70374)
        parser.add_argument("--checkpoint_path", type=str, default=config.CKPT_POINT_DIR)

        args = parser.parse_args()
        return args


if __name__ == '__main__':
    print(config.ROOT_PATH)
    print(config.DATA_DIR)
    config = config()
    args = config.getArgs()

    print(args.max_len)
    print(args.embedding_dim)

    tokenizer, _ = load_tokenizer_binarizer(config.TOKENIZER_BINARIZER)
    print(len(tokenizer.word_index))
    # tokenizer = Tokenizer(num_words = 5, oov_token="")
    print(tokenizer.oov_token)
    # print(tokenizer.word_counts)