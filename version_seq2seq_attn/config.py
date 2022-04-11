import torch
import random
GPU = "cuda:0"
USE_CUDA = torch.cuda.is_available()     # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
device = torch.device(GPU if USE_CUDA else 'cpu')

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)
print(device, "num_cuda: ", NUM_CUDA)

class config(object):
    """配置参数"""
    def __init__(self):
        self.train_path = 'data/train.pkl'
        self.valid_path = 'data/valid.pkl'
        self.test_path = 'data/test.pkl'
        self.vocab_path = 'data/vocab.pkl'
        self.extend_vocab_path = 'data/extend_vocab.pkl'
        self.stopwords_path = 'data/stopword.txt'
        self.pretrained_vector_path = 'data/sgns.sogou.word'
        self.spc_token = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        self.max_len_text = 600
        self.max_len_news = 100
        
        self.use_pointer_gen = True
        self.is_coverage = True