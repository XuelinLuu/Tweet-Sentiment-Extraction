import os
import tokenizers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 5
MODEL_PATH = "model.bin"
BERT_PATH = "bert-base-uncased"
TRAIN_FILES = "../input/train.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True
)