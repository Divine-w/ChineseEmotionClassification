import os
import random
import collections
import torch
import torchtext.vocab as Vocab
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torch import nn
from utils import Trainer, load_pretrained_embedding
from models import BiRNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    data = []
    sample_num = 25000

    with open('./data/positive_process.data', 'r', encoding='utf-8') as f:
        sentences = f.readlines()
        for sentence in sentences[:sample_num]:
            words = [x for x in sentence.strip().split('\t')]
            data.append([words, 0])

    with open('./data/negative_process.data', 'r', encoding='utf-8') as f:
        sentences = f.readlines()
        for sentence in sentences[:sample_num]:
            words = [x for x in sentence.strip().split('\t')]
            data.append([words, 1])

    random.shuffle(data)
    return data


def get_vocab(data):
    tokenized_data = [words for words, _ in data]
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=5)


def preprocess(data, vocab):
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = [words for words, _ in data]
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


batch_size = 64
train_data, test_data = train_test_split(load_data(), test_size=0.2)
vocab = get_vocab(train_data)
print('# words in vocab:', len(vocab))
train_set = Data.TensorDataset(*preprocess(train_data, vocab))
test_set = Data.TensorDataset(*preprocess(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

embed_size, num_hiddens, num_layers = 300, 300, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)

cache = '.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
glove_vocab = Vocab.Vectors(name='./data/sgns.weibo.bigram-char', cache=cache)
net.embedding.weight.data.copy_(
    load_pretrained_embedding(vocab.itos, glove_vocab))
net.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它

lr, num_epochs = 0.01, 5
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
trainer = Trainer(net, loss, optimizer)
trainer.train(train_iter, test_iter, device, num_epochs)
