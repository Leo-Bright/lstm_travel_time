import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

step = 0.15
steps = 750
x = np.arange(0, steps, step)
data = np.sin(x)
print(data)
SEQ_LENGTH = 100
sequence_length = SEQ_LENGTH + 1
result = []

for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
result = np.array(result)

row = round(0.9 * result.shape[0])

train = result[:int(row), :]

np.random.shuffle(train)
x_train = train[:, :-1]
y_train = train[:, -1]
x_test = result[int(row):, :-1]
y_test = result[int(row):, -1]

# LSTM层的输入必须是三维的
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_train)

# Neural Network model
HIDDEN_DIM = 512
LAYER_NUM = 10
model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss="mse", optimizer="rmsprop")
model.summary()
BATCH_SIZE = 32
epoch = 1
model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=epoch, validation_split=0.05)

# start with first frame
curr_frame = x_test[0]

# start with zeros
# curr_frame = np.zeros((100,1))

predicted = []
for i in range(len(x_test)):
    predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
    curr_frame = curr_frame[1:]
    curr_frame = np.insert(curr_frame, [SEQ_LENGTH - 1], predicted[-1], axis=0)
predicted1 = model.predict(x_test)
metri = model.evaluate(x_test, y_test)
predicted1 = np.reshape(predicted1, (predicted1.size,))

plt.figure(1)
plt.subplot(211)
plt.plot(predicted)
# plt.plot(y_test)
plt.subplot(212)
plt.plot(predicted1)
plt.plot(y_test)
plt.show()
