import torch
from torch import nn
from torch.nn import functional as F
import random
from config import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = config()
UNK, PAD, BOS, EOS = '<unk>', '<pad>', '<bos>', '<eos>'

class Encoder(nn.Module):
    """单层双向GRU
    ！！优化：n_layers增加
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1,
                 dropout=0.0, use_pretrained_embeddings=True, pre_embeddings=None):
        super(Encoder,self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        if use_pretrained_embeddings:
            self.embedding = nn.Embedding.from_pretrained(pre_embeddings)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.n_layers, bidirectional=True)
        
        self.ln = nn.LayerNorm(self.hidden_size, eps=1e-5, elementwise_affine=True)
    
    def forward(self, inputs, init_hidden):
        """
        :parma inputs: [batch, seq_len]
        :parma init_hidden: None
        """
        # [batch, seq_len, embed_size] -> [seq_len, batch, embed_size], 适应GRU的输入
        inputs = self.ln(self.embedding(inputs).permute(1,0,2))
        # outputs: [seq_len, batch, hidden_size*num_direction], hidden_states: [n_layers*num_direction, batch, hidden_size]
        outputs,hidden_states = self.gru(inputs,init_hidden)
        outputs = outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
        outputs = outputs.permute(1,0,2) # [batch, seq_len, embed_size]
        hidden_states = hidden_states[:1,:,:] + hidden_states[1:,:,:]
        return outputs, hidden_states
    
    def get_init_hidden(self):
        return None

class Decoder(nn.Module):
    """双层单项GRU
    !!优化：双线性层输出；copy机制；目标函数优化
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.1,
                 use_pretrained_embeddings=True, pre_embeddings=None, use_pointer_gen=True, is_coverage=True):
        super(Decoder,self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_pointer_gen = use_pointer_gen
        
        self.attention = Attention(config.max_len_text, self.hidden_size, is_coverage=is_coverage)
        
        if use_pretrained_embeddings:
            self.embedding = nn.Embedding.from_pretrained(pre_embeddings)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        # 衡量一个词是生成的还是复制的
        if self.use_pointer_gen:
            self.p_gen_sig = nn.Sequential(
                nn.Linear(self.hidden_size*2+self.embed_size, 1),
                nn.Sigmoid()
            )
            
        self.ln = nn.LayerNorm(self.hidden_size, eps=1e-5, elementwise_affine=True)
        # 解码器的输入时间步拼接上下文向量，过一个线性映射
        self.x_context = nn.Linear(self.hidden_size+self.embed_size, self.embed_size)
        # 对输入向量进行解码
        self.gru = nn.GRU(self.embed_size, self.hidden_size, self.n_layers)
        # 将解码状态St和上下文向量ht*拼接后经过两层线性层得到单词表分布P_vocab
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.vocab_size)
        )
        
    def forward(self, inputs, dec_hidden, enc_outputs, enc_padding_mask, context_vector_t1,
                enc_batch_extend_vocab, extra_zeros, coverage):
        """
        :parma inputs: 时间步t的输入（训练时是y_true的输入，预测时是上一时间步的输出） [batch, 1]
        :parma dec_hidden: 当前时间步的隐藏状态 [n_layers, batch, hidden_size]
        :parma enc_outputs: encoder的所有时间步的隐藏状态 [batch, seq_len, hidden_size]
        :parma enc_padding_mask: [batch, seq_len]
        :parma context_vector_t1: 时间步t-1的上下文向量 [batch, hidden_size]
        :parma enc_batch_extend_vocab: 扩充单词表 [batch, seq_len]
        :parma extra_zeros: [batch, extend_vocab_size]
        :parma coverage: 用先前的注意力权重影响当前注意力权重的决策 [batch, seq_len]
        """
        
        x = self.ln(self.embedding(inputs).squeeze(1)) # batch * embed_size
        x = self.x_context(torch.cat((x,context_vector_t1),dim=1)) # batch * embed_size
        # dec_output: [1, batch, hidden_size], s_t: [n_layers, batch, hidden_size]
        dec_output,s_t = self.gru(x.unsqueeze(0), dec_hidden)
        dec_output, s_t = dec_output[0], s_t[-1]
        
        # h_t: batch * hidden_size, atten_dist: batch * vocab_size, coverage_next: batch * vocab_size
        h_t,atten_dist,coverage_next = self.attention(enc_outputs, s_t, enc_padding_mask, coverage)
        
        if self.training and config.is_coverage:
            coverage = coverage_next
        
        p_gen = None
        if self.use_pointer_gen:
            p_gen_input = torch.cat((h_t, s_t, x),dim=1) # batch * hidden_size*3
            p_gen = self.p_gen_sig(p_gen_input).clamp(min=1e-8)
        
        s_t_h_t = torch.cat((dec_output,h_t), dim=1) # batch * hidden_size*2
        vocab_dist = F.softmax(self.out(s_t_h_t), dim=1)
        
        if self.use_pointer_gen:
            vocab_dist_ = p_gen * vocab_dist # batch * vocab_size
            atten_dist_ = (1-p_gen) * atten_dist # batch * vocab_size
            
            if extra_zeros is not None:
                vocab_dist_ = torch.cat((vocab_dist_,extra_zeros), dim=1)
                
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, atten_dist_).clamp(min=1e-8)
        else:
            final_dist = vocab_dist
        
        if config.is_coverage:
            coverage_loss = torch.sum(torch.min(atten_dist, coverage), 1)
        else:
            coverage_loss=None
        return torch.log(final_dist), s_t.unsqueeze(0), h_t, coverage, coverage_loss
    
    def get_init_hidden(self, enc_hidden):
        # 直接使用encoder端输出的隐含向量作为decoder端的初始化
        # enc_hidden: [n_layers*num_direction, batch, hidden_size]
        return enc_hidden
    
class Attention(nn.Module):
    def __init__(self, enc_seq_len, hidden_size, is_coverage=True):
        super(Attention,self).__init__()
        
        self.enc_seq_len = enc_seq_len
        self.hidden_size = hidden_size
        self.is_coverage = is_coverage
        
        if is_coverage:
            self.W_c = nn.Linear(1, self.hidden_size, bias=False)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_s = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, 1, bias=False)
        
    def forward(self, enc_output, dec_hidden, enc_padding_mask, coverage):
        """
        :parma enc_outputs: [batch, seq_len, hidden_size]
        :parma dec_hidden: [batch, hidden_size]
        :parma enc_padding_mask: [batch, seq_len]
        :parma coverage: [batch, seq_len]
        """
        batch_size,seq_len,enc_hidden_size = enc_output.size()
        
        # [batch*seq_len, hidden_size]
        enc_outputs = enc_output.contiguous().view(-1,enc_hidden_size)
        enc_feature = self.W_h(enc_outputs)
        dec_feature = self.W_s(dec_hidden) # [batch, hidden_size]
         # [batch*seq_len, hidden_size]
        dec_feature = dec_feature.unsqueeze(1).expand(batch_size, seq_len, enc_hidden_size).contiguous()
        dec_feature = dec_feature.view(-1,enc_hidden_size)
        
        atten_feature = enc_feature + dec_feature
        if self.is_coverage:
            coverage_input = coverage.view(-1, 1) # [batch*seq_len, 1]
            coverage_feature = self.W_c(coverage_input)
            atten_feature = atten_feature + coverage_feature
        
        # 注意力分数计算
        e_t = self.V(torch.tanh(atten_feature)).view(batch_size,-1) # [batch, seq_len]
        atten_dist = F.softmax(e_t + enc_padding_mask, dim=1)
        # 时间步的atten分数乘以其隐向量，相加得到上下文向量
        context_vector = torch.bmm(atten_dist.unsqueeze(1),enc_output).squeeze(1) # batch * hidden_size
        atten_dist = atten_dist.squeeze(1)
        
        if self.is_coverage:
            coverage = coverage + atten_dist

        return context_vector, atten_dist, coverage
    
class PointerGenerator(nn.Module):
    def __init__(self, encoder, decoder, extend_vocab_size, hidden_size):
        super(PointerGenerator, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.extend_vocab_size = extend_vocab_size
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True)
        
    def forward(self, inputs, target, extend_token_id):
        """
        :parma inputs: [batch, seq_len]
        :parma target: [batch, target_seq_len]
        :parma extend_token_id: [batch, seq_len]
        """
        batch_size,seq_len = inputs.size()
        null_enc_state = self.encoder.get_init_hidden()
        enc_output,enc_hidden  = self.encoder(inputs, null_enc_state)
        
        dec_hidden = self.decoder.get_init_hidden(enc_hidden)
        dec_input = torch.tensor([config.spc_token[BOS]]*batch_size, dtype=torch.long, device=device)
        
        enc_padding_mask = []
        for item in inputs:
            enc_padding_mask.append([float('-inf') if i==0 else 0 for i in item]) # 0为pad id
        enc_padding_mask = torch.tensor(enc_padding_mask, dtype=torch.float32, device=device)
        
        context_vector_t1, enc_batch_extend_vocab, extra_zeros, coverage = self.get_init_input_batch(extend_token_id, batch_size, enc_hidden.size(-1), seq_len)
        
        batch_output = torch.Tensor().to(device) # [batch, target_seq_len, vocab_size]
        for y in target.permute(1,0): # y: seq_len * batch
            dec_hidden, context_vector_t1 = self.ln(dec_hidden), self.ln(context_vector_t1)
            dec_output, dec_hidden, context_vector_t1, coverage, coverage_loss = self.decoder(
                dec_input, dec_hidden, enc_output, enc_padding_mask,
                context_vector_t1, enc_batch_extend_vocab, extra_zeros, coverage
            )
            batch_output = torch.cat((batch_output,dec_output.unsqueeze(1)), dim=1)
            
            if random.uniform(0, 1) > 0.5 and self.training:
                dec_input = y
            else:
                dec_input = dec_output.argmax(dim=1)
        return batch_output, coverage_loss
    
    def get_init_input_batch(self, extend_token_id, batch_size, hidden_size, seq_len):
        coverage = None
        extra_zeros = None
        enc_batch_extend_vocab = None
        
        context_vector_t1 = torch.zeros(batch_size, hidden_size, device=device)
        
        if config.is_coverage:
            coverage = torch.zeros(batch_size, seq_len, device=device)
        
        if config.use_pointer_gen:
            enc_batch_extend_vocab = extend_token_id
            extra_zeros = torch.zeros(batch_size, self.extend_vocab_size, device=device)
        return context_vector_t1, enc_batch_extend_vocab, extra_zeros, coverage