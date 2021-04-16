## 제일먼저 실행해주어야 하는 MNIST 예측 모델입니다.
## 이를 먼저 학습 시켜주어야 합니다, 코드는 다음과 같습니다.

from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt 
import numpy 
import os
import tensorflow as tf 

# seed의 값을 설정해 줍니다.
# 저같은 비전공자 출신을 위해, 시드의 의미를 알려드리겠습니다.

# 컴퓨터 프로그램에서 발생하는 무작위 수는 사실 엄격한 의미의 무작위 수가 아닙니다.
# 어떤 특정한 숫자를 정해주면, 컴퓨터가 정해진 알고리즘에 의해 마치 난수처럼 보이는 수열을 생성하게 됩니다.
# 바로, 이런 시작 숫자를 시드(seed)라고 합니다.
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

# 데이터를 불러옵니다.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = to_categorical(Y_train) # to_categorical = 범주형으로 지정(괄호안의 값을)
Y_test = to_categorical(Y_test) 

# 컨볼루션 신경망의 설정
input_shape = (28, 28, 1)
model = Sequential() # MLP의 레이어가 순차적으로 쌓일 수 있도록 해주는 함수입니다.

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) # 드롭아웃을 하는 이유는, 과적합 문제를 해결하기 위해서인데 실행할 경우 몇 개의 노드를 죽여 남은 노드들로 하여금 실행하게 합니다.
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30,
                        batch_size=128, verbose=1, callbacks=[early_stopping_callback, checkpointer])


# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))

# 테스트 셋의 오차
y_vloss = history.history['val_loss']

# 학습 셋의 오차
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블로 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()