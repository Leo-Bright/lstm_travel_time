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
    sample = []
    bag_line = False
    for item in all_nodes_time:
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
            yield (sample, target)


def build_model():
    # LSTM层的输入必须是三维的
    # Neural Network model
    _model = Sequential()
    # model.add(Masking(mask_value=0, input_shape=(1000, 128)))
    _model.add(LSTM(100, input_shape=(1000, 128), return_sequences=True))
    _model.add(LSTM(100, return_sequences=False))
    _model.add(Dense(1))
    _model.add(Activation('relu'))
    _model.compile(loss="mae", optimizer="adam")
    _model.summary()
    return _model


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

node_embeddings = extract_embeddings(emb)

samples_targets = extract_samples(samples_in_file, node_embeddings)

model = build_model()

samples = []
targets = []
test_samples = []
test_targets = []
test_result = []
have_test_result = False

samples_count = len(samples_in_file)

train_count = 0

for sample_target in samples_targets:
    (_sample, _target) = sample_target
    train_count += 1
    if train_count <= samples_count * 0.9:
        samples.append(_sample)
        targets.append(_target)
        if len(samples) >= 10080 or train_count >= samples_count * 0.9:
            assert len(samples) == len(targets)
            print('training samples at: ', train_count)
            x_train = np.array(samples)
            y_train = np.array(targets)
            BATCH_SIZE = 32
            epoch = 2
            model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=epoch, validation_split=0.01)
            samples = []
            targets = []

    else:
        test_samples.append(_sample)
        test_targets.append(_target)
    if len(test_samples) >= 10080:
        have_test_result = True
        x_test = np.array(test_samples)
        y_test = np.array(test_targets)
        metri = model.evaluate(x_test, y_test)
        print('test result:', metri)
        test_result.append(metri)
        test_samples = []
        test_targets = []

print(emb)
if not have_test_result and len(test_samples) > 0:
    x_test = np.array(test_samples)
    y_test = np.array(test_targets)
    metri = model.evaluate(x_test, y_test)
    print('Finally result:', metri)
else:
    print('mean test loss:', sum(test_result)/len(test_result))


# start with first frame
# curr_frame = x_test[0]

# start with zeros
# curr_frame = np.zeros((100,1))

# predicted = []

# for i in range(len(x_test)):
#     predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
#     curr_frame = curr_frame[1:]
#     curr_frame = np.insert(curr_frame, [SEQ_LENGTH - 1], predicted[-1], axis=0)

# predicted1 = model.predict(x_test)

# predicted1 = np.reshape(predicted1, (predicted1.size,))

# plt.figure(1)
# plt.subplot(211)
# plt.plot(predicted)
# plt.plot(y_test)
# plt.subplot(212)
# plt.plot(predicted1)
# plt.plot(y_test)
# plt.show()
