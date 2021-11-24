import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

def build_corpus(split, make_vocab=True, data_dir="data"):
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir,split+".char.bmes"),"r", encoding="utf-8") as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    word_lists = sorted(word_lists, key=lambda x:len(x),reverse= True)
    tag_lists = sorted(tag_lists,key= lambda x:len(x),reverse=True)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        word2id['<UNK>'] = len(word2id)
        word2id['<PAD>'] = len(word2id)

        tag2id['<PAD>'] = len(tag2id)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps

class MyDataset(Dataset):
    def __init__(self,datas, tags, word_2_index, tag_2_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index

    def __getitem__(self, index):
        data = self.datas[index]
        tag = self.tags[index]

        data_index = [self.word_2_index.get(i,self.word_2_index["<UNK>"]) for i in data ]
        tag_index = [self.tag_2_index.get(i) for i in tag]

        return data_index, tag_index

    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.datas)

    def batch_data_pro(self,batch_datas):
        global device
        data, tag = [], []
        data_len = []
        for da, ta in batch_datas:
            data.append(da)
            tag.append(ta)
            data_len.append(len(da))
        max_len = max(data_len)
        data = [i + [self.word_2_index["<PAD>"]] * (max_len - len(i)) for i in data]
        tag = [i + [self.tag_2_index["<PAD>"]] * (max_len - len(i)) for i in tag]
        data = torch.tensor(data, dtype=torch.long, device=device)
        tag = torch.tensor(tag, dtype=torch.long, device=device)
        return data,tag,data_len

class MyModel(nn.Module):

    def __init__(self, embedding_num, hidden_num, corpus_num, bi, class_num, pad_index):
        super(MyModel, self).__init__()

        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.corpus_num = corpus_num
        self.bi = bi

        self.embedding = nn.Embedding(corpus_num,embedding_num)
        self.lstm = nn.LSTM(embedding_num,hidden_num,batch_first=True,bidirectional=bi)

        if bi:
            self.classifier = nn.Linear(hidden_num * 2,class_num)
        else:
            self.classifier = nn.Linear(hidden_num, class_num)

        self.cross_loss = nn.CrossEntropyLoss(ignore_index=pad_index)

    def forward(self,data_index,data_len,tag_index = None):
        em = self.embedding(data_index)
        pack = nn.utils.rnn.pack_padded_sequence(em,data_len,batch_first=True)
        output,_ = self.lstm(pack)
        output, lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        pre = self.classifier(output)

        if tag_index is not None:
            loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),tag_index.reshape(-1))
            self.pre = torch.argmax(pre, dim=-1)
            return loss
        else:
            self.pre = torch.argmax(pre, dim=-1)
            return


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_word_lists, train_tag_lists, word_2_index, tag_2_index = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    corpus_num = len(word_2_index)
    class_num = len(tag_2_index)

    train_batch_size = 10
    dev_batch_size = len(dev_word_lists)
    epoch = 100
    lr = 0.001
    embedding_num = 256
    hidden_num = 128
    bi = True

    train_dataset = MyDataset(train_word_lists, train_tag_lists, word_2_index, tag_2_index)
    train_dataloader = DataLoader(train_dataset, train_batch_size, False, collate_fn=train_dataset.batch_data_pro)

    dev_dataset = MyDataset(dev_word_lists, dev_tag_lists, word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, False, collate_fn=dev_dataset.batch_data_pro)

    model = MyModel(embedding_num, hidden_num, corpus_num, bi, class_num, word_2_index["<PAD>"])
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr=lr)

    for e in range(epoch):
        model.train()
        train_pre = []
        for data, tag, da_len in train_dataloader:
            loss = model(data,da_len,tag)
            train_pre.append((model.pre, tag, da_len))
            loss.backward()
            opt.step()
            opt.zero_grad()

        pres = []
        tags = []
        for pre, tag, da_len in train_pre:
            for p, t, d_len in zip(pre.cpu(), tag.cpu(), da_len):
                pres.extend(p[:d_len])
                tags.extend(t[:d_len])

        train_score = f1_score(tags, pres, average="micro")
        for dev_data, dev_tag, dev_da_len in dev_dataloader:
            test_loss = model(dev_data, dev_da_len, dev_tag)
            mask = torch.ne(dev_data, float(word_2_index["<PAD>"])).float()
            #countUNK = (dev_data.shape[0] * dev_data.shape[1]) - (torch.ne(dev_data,float(word_2_index["<UNK>"])).float().sum().item())
            pre_dev = model.pre
            dev_tags = []
            pre_devs = []
            for m in range(mask.shape[0]):
                dev_tags.extend(dev_tag.cpu()[m, :int(mask[m].sum().item())].tolist())
                pre_devs.extend(pre_dev.cpu()[m][:int(mask[m].sum().item())])
            score = f1_score(dev_tags, pre_devs, average="micro")
            print(f'train_score : {train_score:.5f}, dev_score : {score:.5f}')

