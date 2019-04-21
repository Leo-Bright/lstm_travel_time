import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


emb = 'po_random_1280_128d.emb'

samples_file = 'pt_trajectory_node_travel_time.travel'


def extract_embeddings(embeddings_file):
    osmid2embeddings = {}
    with open(embeddings_file, 'r') as emb_file:
        for line in emb_file:
            line = line.strip()
            osmid_vector = line.split(' ')
            osmid, node_vec = osmid_vector[0], osmid_vector[1:]
            if len(node_vec) < 10:
                continue
            osmid2embeddings[osmid] = node_vec
    return osmid2embeddings


def extract_samples(all_nodes_time, osmid_embeddings):
    zero_list = [0 for i in range(128)]
    for item in all_nodes_time:
        bag_line = False
        sample = []
        for node in item[:-1]:
            if node not in osmid_embeddings:
                bag_line = True
                break
            id_embeddings = osmid_embeddings[node]
            tmp_emb = [float(ele) for ele in id_embeddings]
            sample.append(tmp_emb)
        target = item[-1]
        while len(sample) < 1000:
            tmp_zero = zero_list.copy()
            sample.append(tmp_zero)
        if bag_line:
            continue
        else:
            yield (sample, float(target))


def average(losses):
    loss_total = 0
    for loss in losses:
        loss_total += loss
    return loss_total/len(losses)


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden) (batch, hidden) is the shape of inputs
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


model = lstm_reg(128, 100)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


samples_in_file = []
with open(samples_file, 'r') as sam_file:
    line_count = 0
    for line in sam_file:
        line_count += 1
        line = line.strip()
        nodes_time = line.split(' ')
        length = len(nodes_time)
        if 10 > length or length > 1000:
            continue
        samples_in_file.append(nodes_time)
    print('extract samples from file done: ', samples_file)

node_embeddings = extract_embeddings(emb)

samples = []
targets = []
test_samples = []
test_targets = []
test_result = []
have_test_result = False

samples_count = len(samples_in_file)
train_num = round(samples_count * 0.9)
test_num = samples_count - train_num
samples_in_file_train = samples_in_file[:train_num]
samples_in_file_test = samples_in_file[train_num:]

iteration_batch = 512
epoch = 2

# training the model with train data
for epo in range(epoch):
    print("training at epoch :", epo)
    samples_targets = extract_samples(samples_in_file_train, node_embeddings)
    train_count = 0
    for sample_target in samples_targets:
        (_sample, _target) = sample_target
        samples.append(_sample)
        targets.append(_target)
        if len(samples) >= iteration_batch:
            assert len(samples) == len(targets)
            train_count += 1
            x_train = torch.Tensor(samples)
            y_train = torch.Tensor(targets)
            var_x = Variable(x_train)
            var_y = Variable(y_train)
            out = model(var_x)
            loss = criterion(out, var_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            samples = []
            targets = []
            if train_count % 10 == 0:  # 每 10 次输出结果
                print('training samples at: ', train_count * iteration_batch)
                print('Epoch: {}, Loss: {:.5f}'.format(epo + 1, loss.data.item()))


# test the model with test data
samples_targets = extract_samples(samples_in_file_test, node_embeddings)
test_count = 0
all_loss_in_batch = []
for sample_target in samples_targets:
    (_sample, _target) = sample_target
    samples.append(_sample)
    targets.append(_target)
    if len(samples) >= iteration_batch:
        assert len(samples) == len(targets)
        test_count += 1
        x_test = torch.Tensor(samples)
        y_test = torch.Tensor(targets)
        var_x = Variable(x_test)
        var_y = Variable(y_test)
        out = model(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        samples = []
        targets = []
        all_loss_in_batch.append(loss.data.item())

print("mean loss in test:", average(all_loss_in_batch))



