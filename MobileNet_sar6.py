# -*- coding: utf-8 -*-
"""
Created on Tsu Dec 11 21:19:31 2018
DeepSatデータセットを用いたMobileNetの実行．

GoogleColab上での呼び出しを想定．メモリサイズがアレなので
trainデータを呼ばずにtestデータをCVして評価．
TensorFlow Eager Execution + Keras API

@author: mickn-1005
"""

import numpy as np
import tensorflow as tf

tfk = tf.keras      # TensorFlow Keras API

class mobn_sar6():
    """docstring for mobn_sar6."""
    def __init__(self, alpha=0.5, depth_multiplier=1):
        self.prep = tfk.Sequential([
                        tfk.layers.ZeroPadding2D(padding=2, dataformat='channels_last')
                        ])
        self.mobn = tfk.application.mobilenet(include_top=False, weights=None, alpha=alpha,, depth_multiplier=depth_multiplier, input_shape=(32,32,4))
        self.fcop = tfk.Sequential([
                        tfk.layers.Flatten()
                        tfk.layers.Dense(256, activation='relu')
                        tfk.layers.Dropout(0.6)
                        tfk.layers.Dense(6, activation='softmax')
                        ])

    def call(self, x):
        x = self.prep(x)
        x = self.mobn(x)
        x = self.fcop(x)
        return x


if __name__ == '__main__':
    sar6 = loadmat("drive/My Drive/MLSAT/sat-6-full.mat")
    x_test = sar6['test_x']
    y_test = sar6['test_y']
    x_test = x_test.T / 255
    y_test = y_test.T

    model = mobn_sar6()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy']
                  )

    early_stopping = tfk.callbacks.EarlyStopping(patience=5)
    model.fit(x_test, y_test,
                validation_split=0.2,
                batch_size=128,
                epochs=1,
                callbacks=[early_stopping],
                verbose=1
                )
