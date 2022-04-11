import pandas as pd
import numpy as np
import json
import re
import os
import pickle

def init():
    from config import config
    import pkuseg
    global seg,config
    seg = pkuseg.pkuseg()
    config = config()
init()

def tokenizer(sentence):
    return [token for token in seg.cut(sentence)]

def get_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        words = f.readlines()
    return [word.strip() for word in words]

def commentary_preprocess(live_content):
    """
    :param live_content: list[n][3]
    """
    time_id, score_id, text_id=-1,-1,-1
    for i in range(3):
        if re.match(r"(\w{,3}\d+[″'])|(^上半场$)",live_content[1][i]):
            time_id = i
        elif re.match('\d-\d',live_content[1][i]):
            score_id = i
        else:
            text_id = i

    # 文本去除重复句子，时间修复，去掉分数列
    ans = []
    up,down=0,0 # 记录上下半场的开始和结束
    idx=0 # 记录上半场的结束时间
    for line in live_content:
        idx+=1
        if re.search(r'裁判哨响！*$',line[text_id]): continue
        line[text_id] = line[text_id].strip('.')
        if not up and re.search('上半场(\w{,2})?结束',line[text_id]):
            line[time_id] = '中场'
            pretime = re.match(r'\d{1,2}',ans[-1][0]) or re.match(r'\d{1,2}',ans[-2][0])
            pretime = int(pretime.group()[0])
            up = 1
        if not down and re.search('下半场(\w{,2})?开始',line[text_id]):
            line[time_id] = "下半场1'"
            down = 1
        
         # 时间修复
        time = re.match(r"(\D{,3})(\d+)[″']",line[time_id])
        time = time.groups() if time else []
        if len(time)==1: # such as 10'，统一成这种格式
            line[time_id] = time[0]+"'"
        elif len(time)==2 and up==down: # such as 下半场10'
            if '上' in time[0] or not up:
                line[time_id] = time[1]+"'"
            else:
                line[time_id] = str(pretime+int(time[1]))+"'"
        else: # 中场、上半场、下半场、''
            if not line[time_id] or '中场' in line[time_id] or (up==1 and down==0):
                line[time_id] = '中场'
                # 中场只保留一行
                if ans[-1][0]=='中场': continue
            else:
                # 修复缺失的时间
                if not up:
                    idx=np.minimum(idx,45)
                    line[time_id] = str(idx)+"'"
                else:
                    idx=np.minimum(idx,95)
                    line[time_id] = str(idx)+"'"
        
        if not ans:
            ans.append([line[time_id],line[text_id]])
            continue
        
        # 同时间合并
        if line[time_id]!=ans[-1][0]:
            res = ""
            tmp = ('，'.join(ans[-1])+'。')
            # 去除重复的句子
            tmp_list = re.split(',|，|\.',tmp)
            for t in tmp_list:
                if t in res: continue
                res += t+'，'
            ans[-1] = res.strip('，')
            ans.append([line[time_id],line[text_id]])
        else:
            ans[-1][1] += '，'+line[text_id]
    if isinstance(ans[-1],list):
        res = ""
        tmp = ('，'.join(ans[-1])+'。')
        tmp_list = re.split(',|，|\.',tmp)
        for t in tmp_list:
            if t in res: continue
            res += t+'，'
        ans[-1] = res.strip('，')
    return ''.join(ans)

def separate_text(text, stopwords):
    """数据预处理：清洗分词去停用词"""
    text_token = []
    for sentence in text:
        tokens = tokenizer(sentence)
        tokens = [token for token in tokens if token not in stopwords]
        text_token.append(tokens)
    return text_token

def preprocess_text(data_path,stopwords_path):
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            datas = pickle.load(f)
        return datas['text'],datas['news']
    else:
        with open(re.sub('(?<=\.).+','json',data_path),'r') as f:
            datas = json.load(f)
        # data cleaning
        text = []
        news = []
        for data in datas:
            text.append(commentary_preprocess(data['commentary']))
            news.append(data['news'])
    
    # 分词
    stopwords = get_stopwords(stopwords_path)
    process_text = separate_text(text,stopwords)
    process_news = separate_text(news,stopwords)

    with open(data_path, 'wb') as f:
        pickle.dump({'text':process_text, 'news':process_news}, f)
    return process_text,process_news

if __name__ == '__main__':
    import sys 
    sys.path.append("..")
    
    train_text, train_news = preprocess_text(config.train_path, config.stopwords_path)
    valid_text, valid_news = preprocess_text(config.valid_path, config.stopwords_path)
    test_text, test_news = preprocess_text(config.test_path, config.stopwords_path)
    print(train_text[0],'\n',train_news[0])