from transformers import BertTokenizer, BertModel, BertConfig
import torch
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler
from torchcrf import CRF
import os
import numpy as np
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ('[PAD]', 0), ('[UNK]', 100), ('[CLS]', 101), ('[SEP]', 102), ('[MASK]', 103)

parameter = {
    # 此处embedding维度为768
    'd_model': 768,
    # 此处的maxlen为512
    'max_len': 512,
    # 训练的批次为10轮
    'epoch': 2,
    # 单次训练的batch_size为1条数据
    'batch_size': 2,
    # 设置dropout，为防止过拟合
    'dropout': 0.1,
    # 配置cpu、gpu
    'device': device,
    # 设置训练学习率
    'lr': 3e-5,
    # 优化器的参数，动量主要用于随机梯度下降
    'momentum': 0.99,
    # 学习策略调整系数gamma
    'gamma': 0.1,
    #预训练模型根路径
    'model_name': 'hfl/chinese-roberta-wwm-ext',
    #配置文件路径
    'config_path': 'config.json',
    #训练数据路径
    'data_path': 'task1_train.txt',
    #是否使用crf
    'crf' : True
}
#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained(parameter['model_name'])

def build_corpus(path=parameter['data_path']):
    with open(path,'r',encoding='utf-8') as f:
        datas = f.readlines()
        train_datas,test_datas = train_test_split(datas,train_size=0.8,shuffle=False)
        train_data_set, train_label2idx, train_idx2label = make_labels(train_datas)
        test_data_set, test_label2idx,_ = make_labels(test_datas)
        assert len(train_label2idx) == len(test_label2idx)
    return train_data_set,test_data_set,train_idx2label,train_label2idx,len(train_label2idx)

def make_labels(datas):
    label_table = {}
    data_set = []
    for data in datas:
        data = json.loads(data)
        text = list(data['originalText'])
        labels = data['entities']
        # 初始化标准ner标签
        label_new = ['O'] * len(text)
        label_table['O'] = 0
        for dicts in labels:
            start_id = dicts['start_pos']
            end_id = dicts['end_pos']
            label_type = dicts['label_type']
            if end_id - start_id == 1:
                type_list = ['B-' + label_type]
                label_new[start_id] = type_list[0]
            elif end_id - start_id > 1:
                type_list = ['B-' + label_type, 'I-' + label_type]
                label_new[start_id] = type_list[0]
                label_new[start_id+1:end_id] = [type_list[1]] * (end_id - start_id-1)
            else:
                raise Exception("标注有误")
            for type in type_list:
                # 为了后面标签转id，提前准好相应的字典
                label_table[type] = label_table.get(type,0) + 1
        # 保存原始的文本和处理好的标签
        data_set.append([text, label_new])
    # 保存标签转id，id转标签的字典
    label2idx = dict(zip(label_table.keys(), range(len(label_table))))
    idx2label = [key for key in label_table.keys()]
    return data_set,label2idx,idx2label

class MyDataset(Dataset):
    def __init__(self,datas,label2idx,device):
        self.datas = datas
        self.label2idx = label2idx
        self.device = device

    def __getitem__(self, index):
        data = self.datas[index]
        token = data[0]
        label = data[1]

        return token, label

    def __len__(self):
        return len(self.datas)

    def batch_data_pro(self,batch_datas):
        tokens, labels = [], []
        data_lens = []
        for x,y in batch_datas:
            if len(x) > (parameter['max_len']-2):
                y = y[:parameter['max_len']-2]
            tokens.append(x)
            labels.append(y)
            data_lens.append(len(x))
        max_len = min(parameter['max_len']-2,max(data_lens))
        data = []
        attention_mask = []
        for i in range(len(tokens)):
            #['[CLS]'] + token + ['[SEP]']
            outs = tokenizer.encode_plus(tokens[i], add_special_tokens=True, padding="max_length", truncation=True,
                                                return_tensors='pt', max_length=max_len+2)
            data.append(outs.input_ids.squeeze(0).tolist())
            attention_mask.append(outs.attention_mask.squeeze(0).tolist())
        tag = [[0] + [self.label2idx[i] for i in l] + [0] * (max_len - len(l)+1)  for l in labels]
        data = torch.tensor(data, dtype=torch.long, device=self.device)
        tag = torch.tensor(tag, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        assert data.shape == tag.shape
        return data,tag,attention_mask

config = BertConfig.from_json_file(parameter['config_path'])

# 方法与bert没有什么区别，只是加上了CRF进行处理
# 构建基于bert+crf实现ner
class Bert_Crf(BertModel):
    def __init__(self, config, parameter):
        super(Bert_Crf, self).__init__(config)
        self.bert = BertModel.from_pretrained(parameter['model_name'])
        embedding_dim = parameter['d_model']
        output_size = parameter['output_size']
        self.fc = nn.Linear(embedding_dim, output_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(output_size,batch_first=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        last_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(last_hidden_states[0])
        logits = self.fc(sequence_output)
        return logits

def ner_predict(val_txt,model,flag,idx2label):
    text_list = list(val_txt)
    max_len = min(parameter['max_len'] - 2, len(val_txt))
    token = tokenizer.encode_plus(text_list, add_special_tokens=True, padding="max_length", truncation=True,
                                       return_tensors='pt', max_length=max_len + 2)
    if flag:
        predict = model.crf.decode(model(torch.tensor(token.input_ids,dtype=torch.long)))[0]
        result = decode(val_txt,predict,idx2label)
    else:
        predict = torch.argmax((model(torch.tensor(token.input_ids,dtype=torch.long))),-1)[0].tolist()
        result = decode(val_txt,predict, idx2label)
    return result

def decode(val_txt,predict,idx2label):
    result_list = [idx2label[p] for p in predict][1:-1]
    keys_list = []
    for ind, i in enumerate(result_list):
        if i == 'O':
            continue
        if i[0] == 'B':
            keys_list.append([val_txt[ind], [i], [ind]])
            continue
        if i[0] == 'I':
            if len(keys_list) > 0  and keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += val_txt[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
            else:
                if len(keys_list) > 0:
                    del keys_list[-1]
            continue
    keys_list = [i[0] + str(i[2]) + '-' + i[1][0][2:] for i in keys_list]
    return keys_list

if __name__ == '__main__':
    # ================================训练==================================#
    train_datas, test_datas, idx2label, label2idx, output_size = build_corpus()
    parameter['output_size'] = output_size
    train_dataset = MyDataset(train_datas,label2idx,device=parameter['device'])
    train_dataloader = DataLoader(train_dataset, parameter['batch_size'], True, collate_fn=train_dataset.batch_data_pro)
    model = Bert_Crf(config,parameter)
    model.to(parameter['device'])
    # 确定训练的优化器和学习策略
    optimizer = optim.AdamW(model.parameters(), lr=parameter['lr'])
    train_steps_per_epoch = len(train_dataset) // parameter['batch_size']
    scheduler = lr_scheduler.StepLR(optimizer, step_size=train_steps_per_epoch, gamma=parameter['gamma'])
    # 确定损失
    criterion = nn.CrossEntropyLoss()
    model.train()
    for e in range(parameter['epoch']):
        for data, tag, attn_mask in train_dataloader:
            out = model(data,attention_mask=attn_mask)
            if parameter['crf']:
                loss = -model.crf(out, tag)
            else:
                loss = criterion(out.view(-1, parameter['output_size']), tag.view(-1))
            optimizer.zero_grad()
            loss.backward()
            # 适当梯度修饰
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
            # 优化器和学习策略更新
            optimizer.step()
            scheduler.step()
            print(loss.item())
    if parameter['crf']:
        torch.save(model.state_dict(),'bert-crf.pth')
    else:
        torch.save(model.state_dict(), 'bert.pth')
    #================================测试==================================#
    model.eval()
    parameter['device'] = 'cpu'
    model.to(parameter['device'])
    test_dataset = MyDataset(test_datas, label2idx,device=parameter['device'])
    test_dataloader = DataLoader(test_dataset, 1, False, collate_fn=test_dataset.batch_data_pro)
    count_table = {l: {'pred':0,'real':0,'tp':0} for l in idx2label if l != 'O'}
    if parameter['crf']:
        model.load_state_dict(torch.load('bert-crf.pth'))
        for test_data, test_tag, test_attn in test_dataloader:
            pred = model(test_data, attention_mask=test_attn)
            pre_index = np.array(model.crf.decode(pred))
            targets = test_tag.cpu().numpy()
            right = (targets == pre_index)
            # 此处注意，回忆一下精确度和召回率的定义；
            # 精确度是，大致可以描述为，判断正确的正例/预测中总共判断正例的数量
            # 召回率是，大致可以描述为，判断正确的正例/实际中总共正例的数量
            for i in range(1, len(idx2label)):
                count_table[idx2label[i]]['pred'] += len(pre_index[pre_index == i]) # i标签下的，tp+fp，预测总正例
                count_table[idx2label[i]]['real'] += len(targets[targets == i]) # i标签下的，tp+fn，实际总正例
                count_table[idx2label[i]]['tp'] += len(targets[right & (targets == i)]) # i标签下的tp
    else:
        model.load_state_dict(torch.load('bert.pth'))
        for test_data,test_tag,test_attn in test_dataloader:
            pred = model(test_data, attention_mask=test_attn)
            pre_index = torch.argmax(pred, -1)
            pre_index = pre_index.reshape(test_data.shape)
            targets = test_tag
            right = (targets == pre_index)
            for i in range(1, len(idx2label)):
                count_table[idx2label[i]]['pred'] += len(pre_index[pre_index == i])
                count_table[idx2label[i]]['real'] += len(targets[targets == i])
                count_table[idx2label[i]]['tp'] += len(targets[right & (targets == i)])
    count_names = {}
    # 获取对应标签中文名，和相应统计值，从1开始，为了过滤标签O的统计
    name,count = list(count_table.keys()),list(count_table.values())
    for ind,i in enumerate(name):
        # 'B-*','I-*','E-*','S-*'都可以用'-'分割，合并同样标签的内容
        i = i.split('-')[1]
        # 综合统计
        if i in count_names:
            count_names[i][0] += count[ind]['pred']
            count_names[i][1] += count[ind]['real']
            count_names[i][2] += count[ind]['tp']
        else:
            count_names[i] = [0,0,0]
            count_names[i][0] = count[ind]['pred']
            count_names[i][1] = count[ind]['real']
            count_names[i][2] = count[ind]['tp']
    # 计算总数
    count_names['all'] = [sum([count_names[i][0] for i in count_names]),
                      sum([count_names[i][1] for i in count_names]),
                      sum([count_names[i][2] for i in count_names])]
    name = count_names.keys()
    count_pandas = pd.DataFrame(count_names.values())
    count_pandas.columns = ['pred','real','tp']
    # 基于tp、tp+fn、tp+fp计算相应的p、r以及计算f1；回忆一下f1计算公式：2pr/(p+r)，fn：(1+b^2)/(b^2)*(pr)/(p+r)
    count_pandas['precision'] = count_pandas['tp']/count_pandas['pred']
    count_pandas['recall'] = count_pandas['tp']/count_pandas['real']
    count_pandas['f1'] = 2*count_pandas['precision']*count_pandas['recall']/(count_pandas['precision']+count_pandas['recall'])
    count_pandas.index = list(name)
    print(count_pandas) #两者效果差不多
    #================================预测==================================#
    val_txt1 = '2-年患者从事体力劳动后腰痛，呈持续性胀痛，并伴有放射痛，放射到左下肢疼痛，无肢体麻木、乏力，休息后加重，活动后缓解，患者遂于我院就诊，行“腰椎管减压、腰4/5椎间盘摘除植骨内固定手术”，2-年患者无明显诱因出现颈肩部疼痛，伴双上肢放射痛，无明显心慌、气短，无潮热、盗汗、咳嗽、咳痰，无明显胸腹部束带感，双下肢无“踏棉花样”感，大小便无明显障碍；患者未予以重视，未行正规治疗，1+年患者于我院行“脊柱内固定器取出+腰椎管减压、脊神经神经根松解减压、腰4/5椎间盘摘除术”，出院后患者仍觉腰背部胀痛伴左下肢放射痛，但症状较术前轻。6月前患者无明显诱因出现剑突下隐痛，伴腹胀恶心，与饮食无关，无节律性，无呕吐反酸、无黑便无腹泻，患者未行任**疗。5+月前患者颈部疼痛伴上肢疼痛，遂于当地按摩店行理疗并自服药物（具体不详），症状无明显缓解，患者为求进一步治疗，前来我院，门诊以“颈椎病”收入我科住院治疗。'
    val_txt2 = '患者因宫颈癌于2015-11-06在全麻下行经腹广泛子宫切除+双侧输卵管切除+盆腔淋巴结清扫术。术后病理39974.15：结合40009.15（宫颈）中-低分化鳞状细胞癌，内生浸润型，肿瘤切面积6.5*1cm，浸润深度大于宫颈壁厚度1/2，脉管内未见明显癌栓,阴道断端、左右宫旁、子宫内膜及肌层、双侧输卵管均未查见癌。另送左闭孔淋巴结4枚、右闭孔淋巴结5枚、左髂髂血管淋巴结4枚及右髂血管淋巴结3枚，均未见转移；另送“左髂总及右髂总淋巴结”组织内未查见淋巴结，未查见癌。增殖状态子宫内膜，子宫腺肌病，双侧输卵管充血，未见明显病变，双侧输卵管系膜囊肿。术后16天予多西他赛120mg+卡铂500mg静滴化疗一次，今为再次化疗来院，门诊以“恶性肿瘤术后化疗，宫颈癌”收入院。患者自上次出院以来，尿频尿少，无尿急尿痛，无头晕、乏力，无腹胀腹痛，无异常阴道流血流液，双下肢无水肿。饮食睡眠可，大便成形，性状可，体重无明显减轻。'
    val_txt3 = '患者2015-1-7因“活动后胸闷、气促10天”于我院住院，入院后于1-7胸腔穿刺+胸膜活检术,1-9行博来霉素45mg胸腔注药，1-19因考虑妇科原发性恶性肿瘤可能转入我科。2015-1-27行卵巢癌根治术（全子宫双附件+大网膜+子宫直肠盆底腹膜+左结肠旁沟腹膜+部分膀胱腹膜切除）。术中腹腔内少量淡粉色乳糜样腹水约200毫升，肝膈面、脾膈面未见明显肿瘤结节，肝、脾脏表面未及肿块，大网膜脾曲可及小结节1枚，直径0.5cm。结肠系膜可及小结节2枚，直径0.5-1.0cm。子宫及双附件外观正常，子宫直肠窝、膀胱腹膜、左侧结肠旁沟多发散在粟粒样结节，直径0.3-0.8cm。术后无肉眼残余病灶。术后病理报告提示为高级别浆液性腺癌，鉴于双侧卵巢及输卵管实质均未见肿瘤组织，仅左侧卵巢血管及纤维组织表面见癌组织，考虑本例肿瘤原发于腹膜可能。术后经患者及家属同意后随机分组入组“晚期上皮性卵巢癌腹腔化疗的ii期临床实验，编号195a”。ecog1分。现患者为行化疗，收住院。患者精神、睡眠可，目前半流少渣饮食中，二便如常，体重无明显变化。'
    print(ner_predict(val_txt1,model,parameter['crf'],idx2label))
    print(ner_predict(val_txt2,model,parameter['crf'], idx2label))
    print(ner_predict(val_txt3, model, parameter['crf'], idx2label))
    
