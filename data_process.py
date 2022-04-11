import numpy as np
import json
import re
import os
import pickle
from datasets import Dataset
from ipywidgets import IntProgress

def init():
    from config import config
    from transformers import BertTokenizer
    global config,tokenizer
    config = config()
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
init()

def commentary_cleaning(live_content):
    """
    :param live_content: list[n][3]
    """
    time_id, score_id, text_id=-1,-1,-1
    for i in range(3):
        if re.match(r"(\w{,3}\s*\d+[″'])|(^上半场$)",live_content[1][i]):
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
    ans[-1] += line[score_id]
    return ''.join(ans)

def text_preprocess(data_path):
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            datas = pickle.load(f)
        return datas
    else:
        with open(re.sub('(?<=\.).+','json',data_path),'r') as f:
            datas = json.load(f)
        # data cleaning
        text = []
        news = []
        for data in datas:
            text.append(commentary_cleaning(data['commentary']))
            news.append(data['news'])

    with open(data_path, 'wb') as f:
        pickle.dump({'text':text, 'news':news}, f)
    return {'text':text, 'news':news}

def convert_to_features(example):
    def segment(texts, seq_len):
        # 文本分段，分为上下半场
        input_first,input_second = [],[]
        for text in texts:
            idx = re.search(r'下半(场|时)(\w{,2})?\s*(开始|易边)|下半场，', text)
            if not idx:
                idx = len(text)//2
            else:
                idx = idx.span()[0]
            input_first.append(prefix+text[:idx])
            input_second.append(text[idx:])

        inputs_first = tokenizer(input_first, max_length=seq_len, padding='max_length', truncation=True)
        inputs_second = tokenizer(input_second, max_length=seq_len, padding='max_length', truncation=True)
        return inputs_first,inputs_second
    
    if 't5' in config.model_name:
        prefix = "summarize: "
    else:
        prefix = ""
        
    if config.is_segment:
        inputs,inputs_second = segment(example[config.columns[0]], config.max_input_length//2)
        for key in ['input_ids', 'attention_mask']:
            inputs[key] = [first[:-1]+second[1:] for first,second in zip(inputs[key],inputs_second[key])]
            
        labels,labels_second = segment(example[config.columns[1]], config.max_target_length//2)
        labels['input_ids'] = [first[:-1]+second[1:] for first,second in zip(labels['input_ids'],labels_second['input_ids'])]
    else:
        original_text = [prefix+text for text in example[config.columns[0]]]
        inputs = tokenizer(original_text, max_length=config.max_input_length, padding='max_length', truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example[config.columns[1]], max_length=config.max_target_length,
                               padding='max_length', truncation=True)

    inputs['labels'] = labels['input_ids']
    return inputs

def build_dataset(data):
    dataset = Dataset.from_dict(data)
    dataset = dataset.map(convert_to_features, batched=True)
    dataset.set_format(type='torch', columns = ['input_ids', 'attention_mask', 'labels'])
    return dataset

def get_sentence_prediction(outputs, target=None):
    batch_summary = []
    for summary_ids in outputs:
        batch_summary.append(tokenizer.decode(summary_ids.argmax(dim=1), skip_special_tokens=True))
        
    if target is not None:
        batch_target = []
        for summary_ids in target:
            batch_target.append(tokenizer.decode(summary_ids, skip_special_tokens=True))
        return batch_summary, batch_target
    return batch_summary

if __name__ == '__main__':
    test_data = text_preprocess(config.test_path)
    test_dataset = build_dataset(test_data)
    print(test_dataset)