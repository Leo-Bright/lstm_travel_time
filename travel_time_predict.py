import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Masking
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

r = 4000
samples = []
targets = []
zero_list = [0 for i in range(128)]
print('=============11111111===========')
with open(samples_file, 'r') as sam_file:
    line_count = 0
    for line in sam_file:
        if len(samples) >= r:
            break
        line_count += 1
        if line_count % 1000 == 0:
            print('line_count:', line_count)
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
            tmp_emb = [float(ele) for ele in node_embeddings]
            sample.append(tmp_emb)
        while len(sample) < 1000:
            tmp_zero = zero_list.copy()
            sample.append(tmp_zero)
        if bag_line:
            continue
        else:
            targets.append(nodes_time[-1])
            samples.append(sample)

print('=============22222222===========')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# step = 0.15
# steps = 750
# x = np.arange(0, steps, step)
# data = np.sin(x)
# print(data)
# SEQ_LENGTH = 100
# sequence_length = SEQ_LENGTH + 1
# result = []

# for index in range(len(data) - sequence_length):
#     result.append(data[index: index + sequence_length])

# samples: 1-dimension is sample, 2-dimension is node, 3-dimension is embedding


assert len(samples) == len(targets)

samples = np.array(samples)
targets = np.array(targets)
row = round(0.9 * samples.shape[0])

x_train = samples[:int(row)]
y_train = targets[:int(row)]

x_test = samples[int(row):]
y_test = targets[int(row):]

# LSTM层的输入必须是三维的
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# print(x_train)

# Neural Network model
model = Sequential()
# model.add(Masking(mask_value=0, input_shape=(1000, 128)))
model.add(LSTM(50, input_shape=(1000, 128), return_sequences=False))
# model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss="mae", optimizer="rmsprop")
model.summary()
BATCH_SIZE = 32
epoch = 1
model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=epoch, validation_split=0.05)

# start with first frame
curr_frame = x_test[0]

# start with zeros
# curr_frame = np.zeros((100,1))

predicted = []

# for i in range(len(x_test)):
#     predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
#     curr_frame = curr_frame[1:]
#     curr_frame = np.insert(curr_frame, [SEQ_LENGTH - 1], predicted[-1], axis=0)

predicted1 = model.predict(x_test)
metri = model.evaluate(x_test, y_test)
print(metri)
predicted1 = np.reshape(predicted1, (predicted1.size,))

plt.figure(1)
plt.subplot(211)
plt.plot(predicted)
plt.plot(y_test)
plt.subplot(212)
plt.plot(predicted1)
plt.plot(y_test)
plt.show()
