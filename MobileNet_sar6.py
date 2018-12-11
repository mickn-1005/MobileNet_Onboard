# -*- coding: utf-8 -*-
"""
Created on Tsu Dec 11 21:19:31 2018
DeepSatデータセットを用いたMobileNetの実行．

GoogleColab上での呼び出しを想定．メモリサイズがアレなので
trainデータを呼ばずにtestデータをCVして評価．

@author: mickn-1005
"""

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

tfk = tf.keras      # TensorFlow Keras API

class mobn_sar6(tfk.Model):
    """docstring for mobn_sar6."""
    def __init__(self, alpha=0.5, depth_multiplier=1):
        super(mobn_sar6, self).__init__(name='mobn_sar6')

        self.mobn = tfk.applications.mobilenet.MobileNet(include_top=False,
                                                         weights=None,
                                                         alpha=alpha,
                                                         depth_multiplier=depth_multiplier,
                                                         input_shape=(32,32,4))

    def build(self):
        dsimg = tfk.layers.Input(shape=(28,28,4))
        x = tfk.layers.ZeroPadding2D(padding=2)(dsimg)
        x = self.mobn(x)
        x = tfk.layers.Flatten()(x)
        x = tfk.layers.Dense(256, activation='relu')(x)
        x = tfk.layers.Dropout(0.6)(x)
        x = tfk.layers.Dense(6, activation='softmax')(x)
        model = tfk.Model(inputs=dsimg, outputs=x)
        return model


if __name__ == '__main__':
    sar6 = loadmat("drive/My Drive/MLSAT/sat-6-full.mat")
    x_test = sar6['test_x']
    y_test = sar6['test_y']
    x_test = x_test.T / 255
    x_test = x_test.transpose(0,2,3,1)
    y_test = y_test.T

    mobn = mobn_sar6()
    model = mobn.build()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy']
                  )
    model.summary()

    early_stopping = tfk.callbacks.EarlyStopping(patience=5)
    model.fit(x_test, y_test,
                validation_split=0.2,
                batch_size=128,
                epochs=100,
                callbacks=[early_stopping],
                verbose=1
                )
