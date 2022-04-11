import torch
import numpy as np
UNK, PAD, BOS, EOS = '<unk>', '<pad>', '<bos>', '<eos>'
def build_vocab(processed_text,processed_news,min_freq=3):
    """构建词典"""
    all_content = []
    all_content.extend(processed_text)
    all_content.extend(processed_news)
    
    # 统计词频
    tokens_dict = {}
    for content in all_content:
        for token in content:
            tokens_dict[token] = tokens_dict.get(token,0)+1

    vocab = {}
    extend_vocab = {}
    idx = 4
    # 映射 token and id
    for k,v in tokens_dict.items():
        if v>=min_freq:
            vocab[k] = idx
            idx+=1
        elif v>1:
            extend_vocab[k] = -1
    for k in extend_vocab.keys():
        extend_vocab[k] = idx
        idx+=1
    vocab.update({PAD:0, UNK:1, BOS:2, EOS:3})
    return vocab, extend_vocab

def build_dataset(vocab, extend_vocab, processed_content, max_len, is_extend_vocab=False, sentence_type=None):
    """pad token and to id"""
    content = []
    extend_content = []
    
    for sent in processed_content:
        if sentence_type == "summary":
            # 为摘要加上结尾符
            if len(sent)<max_len:
                sent.extend([EOS]+[PAD]*(max_len-len(sent)))
            else:
                sent[:] = sent[:max_len] + [EOS]
        else:
            sent[:] = sent[:max_len] + [PAD]*np.maximum(max_len-len(sent),0)
        
        if sentence_type == "summary":
            sent_id = [extend_vocab[token] if token in extend_vocab else vocab.get(token,vocab[UNK]) for token in sent]
        else:
            sent_id = [vocab.get(token,vocab[UNK]) for token in sent]
        if is_extend_vocab:
            extend_id = [extend_vocab[token] if token in extend_vocab else vocab.get(token,vocab[UNK]) for token in sent]
                
        content.append(sent_id)
        if is_extend_vocab: extend_content.append(extend_id)
    return torch.tensor(content), torch.tensor(extend_content)

def get_pretrained_embedding(vocab,pretrain_embedding_path,vector_dim=300):
    """加载词向量，vocab的token_id与embedding的index对齐，用作torch.nn.Embedding.from_pretrained(Embedding)"""
    with open(pretrain_embedding_path, 'r+', encoding='utf-8') as f:
        embeddings = torch.rand(len(vocab),vector_dim)
        for _,line in enumerate(f.readlines()):
            if _==0: continue
            line = line.strip().split(' ')
            if line[0] in vocab:
                idx = vocab[line[0]]
                embeddings[idx] = torch.tensor([float(i) for i in line[1:]], dtype=torch.float32)
        for i in range(4):
            embeddings[idx] = torch.ones(vector_dim, dtype=torch.float32)*float(i/100)
    return embeddings