import torch
import time
from tqdm import tqdm


def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])  # 初始化为0
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed


class Trainer:
    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def evaluate_accuracy(self, data_iter, net, device=None):
        if device is None and isinstance(net, torch.nn.Module):
            # 如果没指定device就使用net的device
            device = list(net.parameters())[0].device
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X, y in data_iter:
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
                n += y.shape[0]
        return acc_sum / n

    def train(self, train_iter, test_iter, device, num_epochs):
        net = self.net.to(device)
        loss = self.loss
        optimizer = self.optimizer
        print("training on ", device)
        batch_count = 0
        opt_test_acc = 0
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in tqdm(train_iter):
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            test_acc = self.evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            # if test_acc > opt_test_acc:
            #     opt_test_acc = test_acc
            #     dirName = './models'
            #     if not os.path.exists(dirName):
            #         os.mkdir(dirName)
            #     PATH = dirName + '/textCNN_net.pth'
            #     torch.save(net, PATH)
            #     print('模型保存成功：', PATH)
