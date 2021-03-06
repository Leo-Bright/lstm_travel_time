from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


emb = 'tokyo/line/tk_LINE1_128'

samples_file = 'tokyo/samples/tk_trajectory_transport_all_node_travel_time_450.travel'


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


def extract_samples(all_nodes_time, osmid_embeddings, max_length):
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
        while len(sample) < max_length:
            tmp_zero = zero_list.copy()
            sample.append(tmp_zero)
        if bag_line:
            continue
        else:
            yield (sample, target)


def build_model():
    # LSTM层的输入必须是三维的
    _model = tf.keras.models.Sequential()
    _model.add(tf.keras.layers.LSTM(56, input_shape=(max_sample_length, 128)))
    _model.add(tf.keras.layers.Dense(1))
    _model.compile(loss="mae", optimizer="adam")
    return _model


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

samples_in_file = []
max_sample_length = 0
with open(samples_file, 'r') as sam_file:
    line_count = 0
    for line in sam_file:
        line_count += 1
        line = line.strip()
        nodes_time = line.split(' ')
        length = len(nodes_time)
        if 10 > length or length > 1000:
            continue
        if length > max_sample_length:
            max_sample_length = length
        samples_in_file.append(nodes_time)
    print('extract samples from file done: ', samples_file)

node_embeddings = extract_embeddings(emb)

samples_targets = extract_samples(samples_in_file, node_embeddings, max_sample_length)

model = build_model()

samples = []
targets = []
test_samples = []
test_targets = []
test_result = []
have_test_result = False

samples_count = len(samples_in_file)
train_num = round(samples_count * 0.9)
train_count = 0
iteration_batch = 10080

for sample_target in samples_targets:
    (_sample, _target) = sample_target
    train_count += 1
    if train_count <= train_num:
        samples.append(_sample)
        targets.append(_target)
        if len(samples) >= iteration_batch or train_count == train_num:
            assert len(samples) == len(targets)
            print('training samples at: ', train_count)
            x_train = np.array(samples)
            y_train = np.array(targets)
            BATCH_SIZE = 64
            epoch = 200
            model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=epoch, validation_split=0.01)
            samples = []
            targets = []
    else:
        test_samples.append(_sample)
        test_targets.append(_target)
    if len(test_samples) >= iteration_batch:
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

