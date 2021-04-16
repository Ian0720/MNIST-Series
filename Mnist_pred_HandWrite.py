# 마지막으로, 직접 A4용지에 작성한 숫자를 png파일로 준비한 뒤
# 화면에 뿌려보는 작업을 하려 합니다, 물론 이제 직접 쓴 손글씨 숫자를 인식하는 부분입니다.
# 이것은 되도록이면, 주피터를 이용해서 실행해주세요 (코딩 따라 치시면서 하시면 좋습니다)

import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread("C:/AI/MNIST/Sample.png")

plt.figure(figsize=(15, 12))
plt.imshow(img)

# 먼저 Grayscale로 바꾸어 줄게요
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(15, 12))
plt.imshow(img_gray)

# 다음은, 가우시안 블러를 사용해서 노이즈를 제거해주려 합니다
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
plt.figure(figsize=(15, 12))
plt.imshow(img_blur)

# 이진화를 적용해서 흑백 사진으로 만들어 주겠습니다
ret, img_th = cv2.threshold(img_blur, 100, 230, cv2.THRESH_BINARY_INV)
plt.figure(figsize=(15, 12))
plt.imshow(img_th)

# 다음은, 윤곽선을 검출하는 작업을 수행하겠습니다
contours, hierachy= cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(each) for each in contours]

# 우리가 검출해야할 숫자는, 12자리인데 너무 작거나 혹은 너무 큰 이상한 것들이 많이 찍힘을 확인 가능하실거에요
# 따라서, 이 부분을 정리할 필요가 있어보이죠?

# 컨투어의 결과는 x좌표, y좌표, width, height인데 이를 이용해서 적당한 사각형을 찾는 것이 급선무입니다.
# 해서 넓이와 높이를 곱해서 정렬해볼게요
tmp = [w*h for (x,y,w,h) in rects]
tmp.sort()

# 아주 작은 사각형들과 아주 큰 사각형 2개를 제외하고 12개의 사각형만 남도록 소스를 수정해주어야 합니다.
rects = [(x,y,w,h) for (x,y,w,h) in rects if ((w*h>60000)and(w*h<500000))]

# 여기까지 완료가 되었다면, 딱 12개의 사각형만 남았을거에요
# 원본 이미지를 저장해 놓고
img_for_class = img.copy()

for rect in rects:
# 직사각형을 그려주는 부분이에요!
    cv2.rectangle(img, (rect[0], rect[1]),
                    (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0),5)

plt.figure(figsize=(15,12))
plt.imshow(img)

# 이제 인식할 이미지에 컨투어를 적용할 수 있게 되었어요
# 각각의 이미지를 따로 저장해보겠습니다
img_result = []
margin_pixel = 60

for rect in rects:
    img_result.append(img_for_class[rect[1]-margin_pixel : rect[1]+rect[3]+margin_pixel,
                            rect[0]-margin_pixel : rect[0]+rect[2]+margin_pixel])

# 이제, 컨투어 영역으로 추출한 이미지들을 찍어봐야 합니다
plt.figure(figsize=(8, 6))
count=0

for n in range(12):
    count += 1
    plt.subplot(3, 4, count)
    plt.imshow(img_result[n], cmap='Greys', interpolation='nearest')

plt.tight_layout()
plt.show()

# 이미지 크기를 동일하게 28x28사이즈로 바꾸어줄게요
plt.figure(figsize=(12, 8))
count=0
for n in img_result:
    count += 1
    plt.subplot(3, 4, count)
    plt.imshow(cv2.resize(n,(28,28)), cmap='Greys', interpolation='nearest')

plt.tight_layout()
plt.show()

# 이제 이전에 학습했던 모델을 불러와 보겠습니다
import sys
import tensorflow as tf
import keras

from keras.models import load_model

model = load_model("C:/AI/MNIST/model/13-0.0196.hdf5")
model.summary()

# 컨투어에서 잘라진 이미지를 넣고 테스트 해보겠습니다
# Warning 나오는거 귀찮아서... 생략해주는 코드를 넣을게요!
# 가볍게 숫자 하나만 찍어서 예측해보죠!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

test_num = cv2.resize(img_result[2], (28,28))[:,:,1]
test_num = (test_num < 70) * test_num
test_num = test_num.astype('float32') / 255.

plt.imshow(test_num, cmap='Greys', interpolation='nearest')

test_num = test_num.reshape((1, 28, 28, 1))

print('The Answer is ', model.predict_classes(test_num))

# 이제 전부 돌려보겠습니다, 제 학습기준에는 100%는 안나오네요...
# 완벽을 위해서는 학습을 더 해야하나 봅니다 하핫
count = 0
nrows = 3
ncols = 4

plt.figure(figsize=(12,8))

for n in img_result:
    count += 1
    plt.subplot(nrows, ncols, count)

    test_num = cv2.resize(n, (28,28))[:,:,1]
    test_num = (test_num < 70) * test_num
    test_num = test_num.astype('float32') / 255.

    plt.imshow(test_num, cmap='Greys', interpolation='nearest')
    test_num = test_num.reshape((1, 28, 28, 1))

    plt.title(model.predict_classes(test_num))

plt.tight_layout()
plt.show()