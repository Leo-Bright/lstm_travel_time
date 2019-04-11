import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from numpy import newaxis
import os

emb = 'po_random_1280_128d.emb'

samples_file = 'pt_trajectory_node_travel_time.travel'

osmid_embeddings = {}

with open(emb, 'r') as emb_file:
    for line in emb_file:
        line = line.strip()
        osmid_vector = line.split(' ')
        osmid, node_vec = osmid_vector[0], osmid_vector[1:]
        if len(node_vec) < 10:
            continue
        osmid_embeddings[osmid] = node_vec

samples = []
targets = []

with open(samples_file, 'r') as sam_file:
    for line in sam_file:
        line = line.strip()
        nodes_time = line.split(' ')
        length = len(nodes_time)
        if 10 > length or length > 1000:
            continue
        sample = []
        bag_line = False
        for node in nodes_time[:-1]:
            if node not in osmid_embeddings:
                bag_line = True
                break
            node_embeddings = osmid_embeddings[node]
            sample.append(node_embeddings)

        if bag_line:
            continue
        else:
            targets.append(nodes_time[-1])
            samples.append(sample)

samples = np.array(samples)
targets = np.array(targets)

row = round(0.9 * samples.shape[0])

assert len(samples) == len(targets)

x_train = samples[:int(row), :, :]
y_train = targets[:int(row)]

x_test = samples[int(row):, :, :]
y_test = targets[int(row)]

# data_csv = pd.read_csv('./data.csv', usecols=[1])
#
# # 数据预处理
# data_csv = data_csv.dropna()
# dataset = data_csv.values
# dataset = dataset.astype('float32')
# max_value = np.max(dataset)
# min_value = np.min(dataset)
# scalar = max_value - min_value
# dataset = list(map(lambda x: x / scalar, dataset))
#
#
# def create_dataset(dataset, look_back=2):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back):
#         a = dataset[i:(i + look_back)]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back])
#     return np.array(dataX), np.array(dataY)


# 创建好输入输出
# data_X, data_Y = create_dataset(dataset)

# 划分训练集和测试集，70% 作为训练集
# train_size = int(len(data_X) * 0.7)
# test_size = len(data_X) - train_size
# train_X = data_X[:train_size]
# train_Y = data_Y[:train_size]
# test_X = data_X[train_size:]
# test_Y = data_Y[train_size:]

train_X = x_train
train_Y = y_train

test_X = x_test
test_Y = y_test

train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


net = lstm_reg(2, 4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 开始训练
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data.item()))

net = net.eval() # 转换成测试模式

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data) # 测试集的预测结果

# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()

# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')