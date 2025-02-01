
# CV_Finetuning_seg
Segmentation model Finetuning <br>
[프로젝트설명PPT](https://github.com/LIMSCODE/CV_Finetuning_seg/blob/main/%EC%BB%B4%ED%93%A8%ED%84%B0%EB%B9%84%EC%A0%84%20HW3.pptx)
<br><br>

### [cv_1) 이미지 특징추출후 SVM학습, 예측값생성](https://github.com/LIMSCODE/CV_finetuning/blob/main/cv_1.ipynb)
#### 1-1) 이미지특징 추출 <br>
이미지에서 특징추출 (SIFT descriptor)  <br>
PCA로 정규화된 특징벡터 생성 (features) <br>
#### 1-2) 이미지특징 인코딩(히스토그램화) <br>
특징인코딩 (codebook.encode(features)) <br>
인코딩된 특징벡터 (train_encoded_features)  <br>
#### 1-3) 인코딩된벡터로 SVM학습, Confusion Matrix생성 <br>
인코딩된특징으로 X_train, Y_train생성후  <br>
SVM학습 (model.fit(X_train, Y_train)) <br>
모델예측값생성 (y_pred = mode.predict(X_test)) <br>
Confusion Matrix생성 (Confusion_Matrix = confusion_matrix(y_test, y_pred)) <br>
  
### [cv_2) MLP,CNN구현, 학습, 성능평가](https://github.com/LIMSCODE/CV_finetuning/blob/main/cv_2.ipynb)
#### 2-1 ) EuroSAT 데이터셋전처리 <br>
EuroSAT데이터셋 다운로드, 전처리 <br>
train, val, test 데이터셋으로 분할  <br>
#### 2-2 ) MLP모델구현  <br>
멀티레이어 정의 <br>
Activation(비선형성추가), Dropout(과적합방지) 사용 <br>
#### 2-3 ) CNN모델구현 <br>
컨볼루션레이어 정의 (특징추출) <br>
Activation(ReLU 활성화함수), Pooling사용 <br>
Fully Connected레이어 정의 (최종출력생성)  <br>
Activation, Dropout 사용  <br>
#### 2-4 ) MLP,CNN 모델학습, 결과분석 <br>
손실함수 (nn.CrossEntropy) 최소화방향으로  <br>
Adam옵티마이저 적용하여 모델업데이트   <br>
테스트데이터셋으로 성능평가하여 CNN성능이 더좋음을확인  <br>
(MLP:1차원벡터변환으로 공간구조손실 / CNN:공간구조유지한 convolution연산구조) <br>
 
### [cv_3) Segmentation 모델 finetuning, 성능평가](https://github.com/LIMSCODE/CV_finetuning/blob/main/cv_3.ipynb)
#### 3-1 ) Segmentation 모델선택  , Segmentation테스트  <br>
모델 : COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml  <br>
풍선이미지 데이터셋 다운로드 / 전처리 (def get_balloon_dicts) <br>
풍선이미지 데이터셋을 모델에 적용한 segmentation결과확인   <br>
#### 3-2 ) Segmentation 모델 finetuning 
파인튜닝 config 설정 <br>
모델 디렉토리생성 <br>
모델 학습 (trainer = DefaultTrainer(cfg))
#### 3-3 ) 모델 성능평가 (COCOEvaluator사용, AP, AR측정)
파인튜닝된모델 vs 사전학습모델

