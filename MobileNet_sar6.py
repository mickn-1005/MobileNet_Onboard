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
import os
import sys
tfk = tf.keras      # TensorFlow Keras API

def IF_filename_toDrive(filename):
    # Colabでマウントされているドライブとローカル環境下での実行の差分を吸収する関数
    if sys.platform=='linux':   # google colabでの実行時
        filename = 'drive/My Drive/tekitou/' + filename
        file_path = os.path.dirname(filename)
    else:                       # macOS, Windows（GitHubのディレクトリ構造上で実行を行える時）
        pass
    return filename

def savepath_creation(filename):
    file_path = os.path.dirname(filename)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return filename

class mobn_sar6(tfk.Model):
    """docstring for mobn_sar6."""
    def __init__(self, alpha=0.5, depth_multiplier=1):
        super(mobn_sar6, self).__init__(name='mobn_sar6')

        self.widmul = depth_multiplier
        self.mobn = tfk.applications.MobileNet(include_top=False,
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
        self.model = tfk.Model(inputs=dsimg, outputs=x)
        return self.model

    def save_mobn(self):
        # width_multiplierごとにモデルを保存
        self.model.save_weights(savepath_creation('__models/mobn_onboard{}.h5'.format(self.widmul)))

    def load_mobn(self):
        self.model.load_weights('__models/mobn_onboard{}.h5'.format(self.widmul))

if __name__ == '__main__':
    sar6 = loadmat("drive/My Drive/MLSAT/sat-6-full.mat")
    x_test = sar6['test_x']
    y_test = sar6['test_y']
    x_test = x_test.T / 255
    x_test = x_test.transpose(0,2,3,1)
    y_test = y_test.T
    param = np.arange(0.05, 1.01, 0.05)     # width_multiplierについてパラメータを振って挙動を見てみる．

    for widmul in param:
        mobn = mobn_sar6(alpha=widmul)
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
        model.save_mobn()
