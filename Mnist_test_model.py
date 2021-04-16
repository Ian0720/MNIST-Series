from tensorflow.keras.models import load_model
import keras
import numpy as np
import matplotlib.pyplot as plt 
import random

# pred_model 파일에서 학습한 모델을 불러온 후 테스트하는 작업을 시행하려 합니다
# 저같은 경우는, 13번 모델이 학습이 제일 잘되어서 이를 불러오려 해요!
model = load_model('C:/AI/MNIST/model/13-0.0196.hdf5')

# 모델을 요약해 봅니다
model.summary()

# MNIST를 불러와서 정규화를 시켜줄게요
img_rows = 28
img_cols = 28 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_test = x_test.astype('float32') / 255
y_test = keras.utils.to_categorical(y_test, 10)

predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = np.argmax(y_test, axis=1)
good_result = []

# 이제 정확하게 예측한 것만, 화면에 뿌려주는 작업에 들어갈게요
for n in range(0, len(test_labels)):
    if predicted_labels[n] == test_labels[n]:
        good_result.append(n)

samples = random.choices(population=good_result, k=16)

count = 0
nrows = ncols = 4

plt.figure(figsize=(12, 8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(28, 28),
                cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + \
    ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()


# 이번에는 예측하지 못한 것만 화면에 뿌려주는 작업에 들어갈게요
wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

samples = random.choices(population=wrong_result, k=16)

count = 0

nrows = ncols = 4

plt.figure(figsize=(12, 8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(28, 28),
                cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + \
    ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()