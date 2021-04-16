# MNIST-Series 'OpenCV와 Deep Learning'<br/>
<br/>

## 필기체 숫자 영상 인식을 딥러닝을 통해 학습된 모델
- It learned through deep learning of cursive numeric image recognition<br/>
<br/>

## Requirements
- Keras<br/>
- Numpy<br/>
- Matplotlib<br/>
- OpenCV<br/>
- Tensorflow 2.x<br/>
<br/>

## 파일 구성 설명 (File Configuration Description)
|Name|About|Etc|
|:----:|:----:|:----:|
|Dir. model| 딥러닝으로 학습된 모델이 저장된 디렉토리<br/>Directory in which models learned from deep learning are stored|제 노트북 기준, 학습된 모델입니다.<br/>This is a model that was learned based on my laptop.|
|Mnist_pred_model.py|제일 먼저, 시작되어야 하는 파일로 'MNIST' 예측 모델에 해당합니다.<br/>The first model to start with, corresponding to the 'MNIST' prediction model.|해당 소스코드를 실행할 경우, 'model' 디렉토리에 가장 최적의 모델이 저장됩니다.<br/>If you run that source code, the 'model' directory stores the best.|
|Mnist_test_model.py|소스코드에서 학습된 모델 중, 가장 최적화된 모델을 불러오고 테스트 하는 부분입니다.<br/>The part of the model learned from the source code that brings up and tests the most optimized.|저는 13번째 에폭 모델이 최상의 결과이므로, 그것을 가져와서 실행하였습니다.<br/>The 13th epoch model is the best result, I took it and implemented it.|
|Mnist_pred_Handwrite.py|테스트까지 완료한, 학습된 모델에 제가 쓴 손글씨를 입력하여 예측을 진행하였습니다.<br/>I made a prediction by typing my handwriting in the learned model that I completed the test.|가급적이면, 주피터 노트북에서 사용하심을 권장합니다.<br/>If possible, it is recommended that you use it in your Jupiter notebook.|
<br/>

## 소스코드를 사용하여, 예측된 샘플 본(Sample examples predicted using the source code.)
- 예측에 성공한 샘플(Sample which Successful Prediction)<br/>
<img width = 75% src = https://user-images.githubusercontent.com/79067558/114956685-58637f80-9e9a-11eb-86fc-ad22eb425c50.png><br/>
- 예측에 실패한 샘플(Sample with failed Prediction)<br/>
<img width = 75% src = https://user-images.githubusercontent.com/79067558/114957970-0a9c4680-9e9d-11eb-8722-985dcfa6940c.png><br/>
<br/>

## 소스코드를 사용하여, 손글씨를 예측한 샘플 본(Sample 'The HandWrite Image' examples predicted using the source code)
- 예측한 결과물(The result from Handwrite sample image)<br/>
<img width = 75% src = https://user-images.githubusercontent.com/79067558/114958266-9ca44f00-9e9d-11eb-9576-ab41a8529a7c.png><br/>
<br/>

## 작업 과정(Task Course)
- 손글씨 컨투어링 과정(HandWriting Contouring Process)<br/>
<img width = 75% src = https://user-images.githubusercontent.com/79067558/114958411-e7be6200-9e9d-11eb-8ca4-eda3ade142c3.png>
- 28 x 28 사이즈 규격화(Size standardization to 28 x 28)<br/>
<img width = 75% src = https://user-images.githubusercontent.com/79067558/114958583-44218180-9e9e-11eb-94d5-db3387974292.png>
