#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import h5py
import json
import re
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.callbacks import ModelCheckpoint, CSVLogger


# set path
# このファイルのpath
FILE_PATH="/home/uchiumi/mnist-mi/main.py"
# モデルのユニークな名前
MODEL_NAME="mnist_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# logの下に，ディレクトリ: "MODEL_NAME" を作成
LOG_DIR=os.path.join("./log", MODEL_NAME)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)


# 学習に用いたmainファイルをコピー
MODEL_MAIN_PATH    = os.path.join(LOG_DIR, 'main.py')
shutil.copyfile(FILE_PATH, MODEL_MAIN_PATH)
# 学習中，もっとも精度(val acc)の高い重みをH5ファイルで保存
MODEL_WEIGHT_CKP_PATH = os.path.join(LOG_DIR, "best_weights.h5")
# 学習中の各指標をcsvファイルとして保存
MODEL_TRAIN_LOG_CSV_PATH=os.path.join(LOG_DIR, "train_log.csv")

# データ読み込み
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# 正規化
X_train, X_test = X_train / 255.0, X_test / 255.0

# モデル
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28), name='input'),
    
  tf.keras.layers.Dense(256, name='dense_1'),
  tf.keras.layers.Activation(tf.nn.relu, name='relu_1'),
  tf.keras.layers.Dropout(0.2, name='dropout_1'),
    
  tf.keras.layers.Dense(256, name='dense_2'),
  tf.keras.layers.Activation(tf.nn.relu, name='relu_2'),
  tf.keras.layers.Dropout(0.2, name='dropout_2'),
    
  tf.keras.layers.Dense(10, name='dense_3'),
  tf.keras.layers.Activation(tf.nn.softmax, name='softmax')
])

model.summary()

# モデルのコンパイル
model.compile(optimizer=tf.keras.optimizers.SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# set callbacks
callbacks = []
callbacks.append(ModelCheckpoint(MODEL_WEIGHT_CKP_PATH, monitor='val_loss', save_best_only=True, save_weights_only=True))
callbacks.append(CSVLogger(MODEL_TRAIN_LOG_CSV_PATH))

# モデルの学習
history = model.fit(X_train, 
              y_train, 
              batch_size=200, 
              epochs=1,
              verbose=1,
              callbacks=callbacks,
              validation_data=(X_test, y_test))

# 評価
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=1)

print("--------------------------------------")
print("model name : ", MODEL_NAME)
print("validation loss     : {:.5f}".format(val_loss)) 
print("validation accuracy : {:.5f}".format(val_acc)) 


# save plot figure
from utils.plot_log import save_plot_log
save_plot_log(LOG_DIR, MODEL_TRAIN_LOG_CSV_PATH, index='acc')
save_plot_log(LOG_DIR, MODEL_TRAIN_LOG_CSV_PATH, index='loss')
save_plot_log(LOG_DIR, MODEL_TRAIN_LOG_CSV_PATH, index='loss-log10')

# save model "INSTANCE"
ins_name = 'model_instance'
ins_path = os.path.join(LOG_DIR, ins_name) + '.h5'
model.save(ins_path)

# save model "ARCHITECHTURE"
arch_name = 'model_fin_architechture'
arch_path = os.path.join(LOG_DIR, arch_name) + '.json'
json_string = model.to_json()
with open(arch_path, 'w') as f:
    f.write(json_string)

print("---------------------------------------")
print("successfully completed!")
print("Please see more detail log in ", LOG_DIR)



