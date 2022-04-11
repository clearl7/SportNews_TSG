# import torch
# import random
# GPU = "cuda:0"
# USE_CUDA = torch.cuda.is_available()     # 是否使用GPU
# NUM_CUDA = torch.cuda.device_count()
# device = torch.device(GPU if USE_CUDA else 'cpu')

# SEED = 1234
# random.seed(SEED)
# torch.manual_seed(SEED)
# if USE_CUDA:
#     torch.cuda.manual_seed_all(SEED)
# print(device, "num_cuda: ", NUM_CUDA)

class config(object):
    """配置参数"""
    def __init__(self):
        self.train_path = 'data/train.pkl'
        self.valid_path = 'data/valid.pkl'
        self.test_path = 'data/test.pkl'
        self.stopwords_path = 'data/stopword.txt'
	self.train_dataset = 'data/train_dataset.pkl'
        self.valid_dataset = 'data/valid_dataset.pkl'
        self.test_dataset = 'data/test_dataset.pkl'
        
        self.columns = ['text', 'news']
        self.tokenizer_name = "bert-base-chinese"
        self.model_name = "fnlp/bart-base-chinese"
        
        self.is_segment = True
        self.max_input_length = 512*2 # input, source text
        self.max_target_length = 256*2 # summary, target text