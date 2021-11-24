import os
import torch
import torch.nn as nn
from torch.nn import init
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

START_TAG = "<START>"
END_TAG = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"

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

        tag2id["<START>"] = len(tag2id)
        tag2id['<PAD>'] = len(tag2id)
        tag2id["<STOP>"] = len(tag2id)
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
        return data.transpose(0, 1), tag.transpose(0, 1), data_len

class CRFLayer(nn.Module):
    def __init__(self, tag_size):
        super(CRFLayer, self).__init__()
        # transition[i+1][i] 初始化转移矩阵
        self.transition = nn.Parameter(torch.randn(tag_size,tag_size))

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.transition)
        # 将START_TAG, END_TAG初始化成一个足够小的值
        self.transition.detach()[tag_2_index[START_TAG],:] = -10000
        self.transition.detach()[:, tag_2_index[END_TAG]] = -10000

    def forward_allpath(self, feats, mask):
        #前向算法，计算所有路径的得分
        """
        Arg:
          feats: (seq_len, batch_size, tag_size)
          mask: (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        #初始化到达所有标签节点的路径的对数指数得分
        alpha = feats.new_full((batch_size, tag_size), fill_value=-10000)
        #将START_TAG对数置为一个可取初值 log（1）
        alpha[:, tag_2_index[START_TAG]] = 0
        for i, feat in enumerate(feats):
            #广播：(batch_size, next_tag, current_tag)
            #无论标签如何发射得分都是相同的,所以从当前标签广播即可
            emit_score = feat.unsqueeze(-1)  # (batch_size, tag_size, 1)
            #每个样本的转移得分相同,从batchsize维度广播
            transition_score = self.transition.unsqueeze(0)  # (1, tag_size, tag_size)
            #下一个标签的alpha_score相同,从next_tag维度广播
            alpha_score = alpha.unsqueeze(1)  # (batch_size, 1, tag_size)
            alpha_score = alpha_score + transition_score + emit_score
            #从当前标签维度（current_tag）求logsumexp得到next_tag的alpha值,遇到pad则保留最后一个的alpha
            mask_i = mask[i].unsqueeze(-1) #(batch_size,1)
            alpha = torch.logsumexp(alpha_score, -1) * mask_i + alpha * (1 - mask_i)  # (batch_size, tag_size)
        #END_TAG结束
        alpha = alpha + self.transition[tag_2_index[END_TAG]].unsqueeze(0)
        return torch.logsumexp(alpha, -1)  # (batch_size, )

    def score_realpath(self, feats, tags, mask):
        #计算真实路径得分
        """
        Arg:
          feats: (seq_len, batch_size, tag_size)
          tags: (seq_len, batch_size)
          mask: (seq_len, batch_size)
        Return:
          scores: (batch_size, )
        """
        seq_len, batch_size, tag_size = feats.size()
        scores = feats.new_zeros(batch_size)
        #拼接一个START_TAG
        tags = torch.cat([tags.new_full((1, batch_size), fill_value=tag_2_index[START_TAG]), tags],0)  # (seq_len + 1, batch_size)
        for i, feat in enumerate(feats):
            #feat[tag[i+1]] 对齐
            emit_score = torch.stack([f[next_tag] for f, next_tag in zip(feat, tags[i + 1])])
            #self.transition[tag[i+1],tag[i]]
            transition_score = torch.stack([self.transition[tags[i + 1, b], tags[i, b]] for b in range(batch_size)])
            scores += (emit_score + transition_score) * mask[i]
        #self.tarnsitons[self.tag2idx[STOP_TAG],tags[-1]] 拼接end
        transition_to_end = torch.stack([self.transition[tag_2_index[END_TAG], tag[mask[:, b].sum().long()]] for b, tag in enumerate(tags.transpose(0, 1))])
        scores += transition_to_end
        return scores

    def viterbi_decode(self, feats, mask):
        #维特比解码
        """
        :param feats: (seq_len, batch_size, tag_size)
        :param mask: (seq_len, batch_size)
        :return best_path: (seq_len, batch_size)
        """
        seq_len, batch_size, tag_size = feats.size()
        scores = feats.new_full((batch_size, tag_size), fill_value=-10000)
        scores[:, tag_2_index[START_TAG]] = 0
        #记录回溯路径
        pointers = []
        for i, feat in enumerate(feats):
            scores_i = scores.unsqueeze(1) + self.transition.unsqueeze(0) #(batch_size, tag_size(transition), tag_size(scores))
            scores_temp, pointer = torch.max(scores_i, -1) # (batch_size, tag_size)
            scores_next = scores_temp + feat
            pointers.append(pointer)
            mask_i = mask[i].unsqueeze(-1)  # (batch_size, 1)
            scores = scores_next * mask_i + scores * (1 - mask_i)
        pointers = torch.stack(pointers, 0)  # (seq_len, batch_size, tag_size)
        scores += self.transition[tag_2_index[END_TAG]].unsqueeze(0)
        best_score, best_tag = torch.max(scores, -1)  # (batch_size, )
        #反向回溯
        best_path = best_tag.unsqueeze(-1).tolist()  #(batch_size, 1)
        for b in range(batch_size):
            best_tag_b = best_tag[b]
            seq_len_b = int(mask[:, b].sum())
            for ptr_t in reversed(pointers[:seq_len_b, b]):
                # ptr_t shape (tag_size, )
                best_tag_b = ptr_t[best_tag_b].item()
                best_path[b].append(best_tag_b)
            # pop掉START_TAG
            best_path[b].pop()
            best_path[b].reverse()
        return best_path

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_size, hidden_size):
        super(BiLSTMCRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size,padding_idx=word_2_index[PAD])
        self.bilstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_size * 2, tag_size)
        self.crf = CRFLayer(tag_size)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.embedding.weight)
        init.xavier_normal_(self.hidden2tag.weight)

    def get_lstm_features(self, seq, mask):
        # lstm forward 计算发射矩阵
        """
        :param seq: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        embed = self.embedding(seq)  # (seq_len, batch_size, embedding_size)
        embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(0).long().cpu(), enforce_sorted=False)
        lstm_output, _ = self.bilstm(embed)  # (seq_len, batch_size, hidden_size)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
        lstm_features = self.hidden2tag(lstm_output) * mask.unsqueeze(-1)  # (seq_len, batch_size, tag_size)
        return lstm_features

    def neg_log_likelihood(self, seq, tags, mask):
        #对数似然估计 crf loss backward
        """
        :param seq: (seq_len, batch_size)
        :param tags: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        lstm_features = self.get_lstm_features(seq, mask)
        forward_score = self.crf.forward_allpath(lstm_features, mask)
        gold_score = self.crf.score_realpath(lstm_features, tags, mask)
        loss = (forward_score - gold_score).sum()

        return loss

    def predict(self, seq, mask):
        # 预测，维特比解码
        """
        :param seq: (seq_len, batch_size)
        :param mask: (seq_len, batch_size)
        """
        lstm_features = self.get_lstm_features(seq, mask)
        best_paths = self.crf.viterbi_decode(lstm_features, mask)

        return best_paths


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_word_lists, train_tag_lists, word_2_index, tag_2_index = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    vocab_size = len(word_2_index)
    tag_size = len(tag_2_index)

    train_batch_size = 10
    dev_batch_size = len(dev_word_lists)
    epoch = 100
    lr = 0.001
    embedding_num = 256
    hidden_num = 128

    train_dataset = MyDataset(train_word_lists, train_tag_lists, word_2_index, tag_2_index)
    train_dataloader = DataLoader(train_dataset, train_batch_size, False, collate_fn=train_dataset.batch_data_pro)

    dev_dataset = MyDataset(dev_word_lists, dev_tag_lists, word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, False, collate_fn=dev_dataset.batch_data_pro)

    model = BiLSTMCRF(vocab_size, tag_size, embedding_num, hidden_num)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        model.train()
        train_pre = []
        for data, tag, da_len in train_dataloader:
            mask = torch.ne(data, float(word_2_index[PAD])).float()
            loss = model.neg_log_likelihood(data, tag, mask)
            pre = model.predict(data, mask)  #list [batchsize, dalen]
            train_pre.append((pre, tag.transpose(0, 1), da_len))
            loss.backward()
            opt.step()
            opt.zero_grad()

        pres = []
        tags = []
        for pre, tag, da_len in train_pre:
            for p, t, d_len in zip(pre, tag.cpu(), da_len):
                pres.extend(p)
                tags.extend(t[:d_len])

        train_score = f1_score(tags, pres, average="micro")

        model.eval()  # F1,准确率,精确率,召回率
        for dev_data, dev_tag, dev_da_len in dev_dataloader:
            mask = torch.ne(dev_data, float(word_2_index[PAD])).float()
            pre_dev = model.predict(dev_data, mask)
            dev_tags = []
            pre_devs = []
            for m in range(mask.shape[1]):
                dev_tags.extend(dev_tag.cpu()[:int(mask[:, m].sum().item()), m].tolist())
                pre_devs.extend(pre_dev[m])
            score = f1_score(dev_tags, pre_devs, average="micro")
            print(f'train_score : {train_score:.5f}, dev_score : {score:.5f}')