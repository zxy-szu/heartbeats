import os
import math
import time
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import Sequential, utils, regularizers, Model, Input
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout, AvgPool1D
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Flatten, Dense, \
    Dropout, Add

# 加载训练集和测试集(相对路径)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./testA.csv')

# 训练集特征处理与精度量化
train_list = []
for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])
train = pd.DataFrame(np.array(train_list))
train.columns = ['id'] + ['s_' + str(i) for i in range(len(train_list[0]) - 2)] + ['label']  # 特征分离

# 测试集特征处理与精度量化
test_list = []
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])
test = pd.DataFrame(np.array(test_list))
test.columns = ['id'] + ['s_' + str(i) for i in range(len(test_list[0]) - 1)]  # 特征分离

# 查看训练集, 分离标签与样本, 去除 id
y_train = train['label']
x_train = train.drop(['id', 'label'], axis=1)
print(x_train.shape, y_train.shape)

# 查看测试集, 去除 id
X_test = test.drop(['id'], axis=1)
print(X_test.shape)

# 将测试集转换为适应 CNN 输入的 shape
X_test = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
print(X_test.shape, X_test.dtype)

# 使用 SMOTE 对数据进行上采样以解决类别不平衡问题
smote = SMOTE(random_state=2024, n_jobs=-1)
k_x_train, k_y_train = smote.fit_resample(x_train, y_train)
print(f"after smote, k_x_train.shape: {k_x_train.shape}, k_y_train.shape: {k_y_train.shape}")

# 将训练集转换为适应 CNN 输入的 shape
k_x_train = np.array(k_x_train).reshape(k_x_train.shape[0], k_x_train.shape[1], 1)


# 评估函数
def abs_sum(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    loss = sum(sum(abs(y_pred - y_true)))
    return loss


class Net1(K.Model):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(205, 1))
        self.conv2 = Conv1D(filters=32, kernel_size=3, dilation_rate=2, padding='same', activation='relu')
        self.conv3 = Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='same', activation='relu')
        self.conv4 = Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool1 = MaxPool1D(pool_size=3, strides=2, padding='same')
        self.conv5 = Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.conv6 = Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool2 = MaxPool1D(pool_size=3, strides=2, padding='same')
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation='relu')
        self.fc21 = Dense(units=16, activation='relu')
        self.fc22 = Dense(units=256, activation='sigmoid')
        self.fc3 = Dense(units=4, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x1 = self.fc1(x)
        x2 = self.fc22(self.fc21(x))
        x = self.fc3(x1 + x2)
        return x


class GeMPooling(tf.keras.layers.Layer):
    def __init__(self, p=1.0, train_p=False):
        super().__init__()
        self.eps = 1e-6
        self.p = tf.Variable(p, dtype=tf.float32) if train_p else p

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1], keepdims=False)
        inputs = tf.pow(inputs, 1. / self.p)
        return inputs


class Net3(K.Model):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(205, 1))
        self.conv2 = Conv1D(filters=32, kernel_size=3, dilation_rate=2, padding='same', activation='relu')
        self.conv3 = Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='same', activation='relu')
        self.max_pool1 = MaxPool1D(pool_size=3, strides=2, padding='same')
        self.conv4 = Conv1D(filters=64, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.conv5 = Conv1D(filters=128, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool2 = MaxPool1D(pool_size=3, strides=2, padding='same')
        self.conv6 = Conv1D(filters=256, kernel_size=5, dilation_rate=2, padding='same', activation='relu')
        self.conv7 = Conv1D(filters=128, kernel_size=7, dilation_rate=2, padding='same', activation='relu')
        self.gempool = GeMPooling()
        self.dropout1 = Dropout(0.5)
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation='relu')
        self.fc21 = Dense(units=16, activation='relu')
        self.fc22 = Dense(units=256, activation='sigmoid')
        self.fc3 = Dense(units=4, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gempool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x1 = self.fc1(x)
        x2 = self.fc22(self.fc21(x))
        x = self.fc3(x1 + x2)
        return x


class Net8(K.Model):
    def __init__(self):
        super(Net8, self).__init__()
        self.conv1 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(205, 1))
        self.conv2 = Conv1D(filters=32, kernel_size=3, padding='same', dilation_rate=2, activation='relu')
        self.conv3 = Conv1D(filters=64, kernel_size=3, padding='same', dilation_rate=2, activation='relu')
        self.conv4 = Conv1D(filters=128, kernel_size=3, padding='same', dilation_rate=2, activation='relu')
        self.conv5 = Conv1D(filters=128, kernel_size=5, padding='same', dilation_rate=2, activation='relu')
        self.max_pool1 = MaxPool1D(pool_size=3, strides=2, padding='same')
        self.avg_pool1 = AvgPool1D(pool_size=3, strides=2, padding='same')
        self.conv6 = Conv1D(filters=128, kernel_size=5, padding='same', dilation_rate=2, activation='relu')
        self.conv7 = Conv1D(filters=128, kernel_size=5, padding='same', dilation_rate=2, activation='relu')
        self.max_pool2 = MaxPool1D(pool_size=3, strides=2, padding='same')
        self.avg_pool2 = AvgPool1D(pool_size=3, strides=2, padding='same')
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.fc1 = Dense(units=256, activation='relu')
        self.fc21 = Dense(units=16, activation='relu')
        self.fc22 = Dense(units=256, activation='sigmoid')
        self.fc3 = Dense(units=4, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        xm1 = self.max_pool1(x)
        xa1 = self.avg_pool1(x)
        x = tf.concat([xm1, xa1], 2)
        x = self.conv6(x)
        x = self.conv7(x)
        xm2 = self.max_pool2(x)
        xa2 = self.avg_pool2(x)
        x = tf.concat([xm2, xa2], 2)
        x = self.dropout(x)
        x = self.flatten(x)
        x1 = self.fc1(x)
        x2 = self.fc22(self.fc21(x))
        x = self.fc3(x1 + x2)
        return x


# 定义学习率调度器
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.1
    else:
        return lr * 0.01


# 创建学习率调度器回调
lr_callback = LearningRateScheduler(lr_scheduler)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 定义k折交叉验证
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 准备存储每折的结果
val_accuracies = []
Net1_fold_predictions = []
Net3_fold_predictions = []
Net8_fold_predictions = []

# 开始k折交叉验证
for fold, (train_idx, val_idx) in enumerate(kfold.split(k_x_train, k_y_train)):
    print(f"Training fold {fold + 1}")
    # 获取训练和验证集
    x_train_fold, x_val_fold = k_x_train[train_idx], k_x_train[val_idx]
    y_train_fold, y_val_fold = k_y_train[train_idx], k_y_train[val_idx]

    # 模型训练并保存
    Net1_model = Net1()
    Net1_model.compile(optimizer=Adam(),
                       loss=SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])
    Net1_model.fit(x_train_fold, y_train_fold,
                   batch_size=64,
                   epochs=12,
                   callbacks=[lr_callback],
                   validation_data=(x_val_fold, y_val_fold))
    Net1_model.save('Net1_model' + str(fold) + '.h5')

    Net3_model = Net3()
    Net3_model.compile(optimizer=Adam(),
                       loss=SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])
    Net3_model.fit(x_train_fold, y_train_fold,
                   batch_size=64,
                   epochs=12,
                   callbacks=[lr_callback],
                   validation_data=(x_val_fold, y_val_fold))
    Net3_model.save('Net3_model' + str(fold) + '.h5')

    Net8_model = Net8()
    Net8_model.compile(optimizer=Adam(),
                       loss=SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])
    Net8_model.fit(x_train_fold, y_train_fold,
                   batch_size=64,
                   epochs=12,
                   callbacks=[lr_callback],
                   validation_data=(x_val_fold, y_val_fold))
    Net8_model.save('Net8_model' + str(fold) + '.h5')

    predictions_Net1 = Net1_model.predict(X_test)
    Net1_fold_predictions.append(predictions_Net1)

    predictions_Net3 = Net3_model.predict(X_test)
    Net3_fold_predictions.append(predictions_Net3)

    predictions_Net8 = Net8_model.predict(X_test)
    Net8_fold_predictions.append(predictions_Net8)

predictions_weighted = 0.035 * Net1_fold_predictions[0] + 0.031 * Net3_fold_predictions[0] + 0.034 * Net8_fold_predictions[0]
for i in range(1, 10):
    predictions_weighted += 0.035 * Net1_fold_predictions[i] + 0.031 * Net3_fold_predictions[i] + 0.034 * Net8_fold_predictions[i]

# 准备提交结果
submit = pd.DataFrame()
submit['id'] = range(100000, 120000)
submit['label_0'] = predictions_weighted[:, 0]
submit['label_1'] = predictions_weighted[:, 1]
submit['label_2'] = predictions_weighted[:, 2]
submit['label_3'] = predictions_weighted[:, 3]
submit.head()

# 第一次后处理未涉及的难样本 index
others = []

# 第一次后处理 - 将预测概率值大于 0.5 的样本的概率置 1，其余置 0
threshold = 0.5
for index, row in submit.iterrows():
    row_max = max(list(row[1:]))  # 当前行中的最大类别概率预测值
    if row_max > threshold:
        for i in range(1, 5):
            if row[i] > threshold:
                submit.iloc[index, i] = 1  # 大于 0.5 的类别概率预测值置 1
            else:
                submit.iloc[index, i] = 0  # 其余类别概率预测值置 0
        else:
            others.append(index)  # 否则，没有类别概率预测值不小于 0.5，加入第一次后处理未涉及的难样本列表，等待第二次后处理
            print(index, row)

# 第二次后处理 - 在预测概率值均不大于 0.5 的样本中，若最大预测值与次大预测值相差大于 0.04，则将最大预测值置 1，其余预测值置 0；
#                否则，对最大预测值和次大预测值不处理 (难分类)，仅对其余样本预测值置 0
for idx in others:
    value = submit.iloc[idx].values[1:]
    ordered_value = sorted([(v, j) for j, v in enumerate(value)], reverse=True)  # 根据类别概率预测值大小排序
    # print(ordered_value)
    if ordered_value[0][0] - ordered_value[1][0] >= 0.04:  # 最大与次大值相差至少 0.04
        submit.iloc[idx, ordered_value[0][1] + 1] = 1  # 则足够置信最大概率预测值并置为 1
        for k in range(1, 4):
            submit.iloc[idx, ordered_value[k][1] + 1] = 0  # 对非最大的其余三个类别概率预测值置 0
    else:
        for s in range(2, 4):
            submit.iloc[idx, ordered_value[s][1] + 1] = 0  # 难分样本，仅对最小的两个类别概率预测值置 0

    print(submit.iloc[idx])

print(submit.head(5))

# 保存预测结果
submit.to_csv(("./submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), index=False)

