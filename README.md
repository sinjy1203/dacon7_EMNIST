# 월간 데이콘 컴퓨터 비전 학습 경진대회
>글자에 가려진 숫자 이미지 데이터셋을 이용해 숫자를 예측하는 대회

</br>

## 1. 제작 기간 & 참여 인원
- 2020년 08월 03일 ~ 2020년 09월 14일
- 개인으로 참여

</br>

## 2. 사용 기술
- python
- pytorch
- CNN
- ResNet
- image augmentation
- ensemble

</br>

## 3. file 설명
`train.py` training model, visualization with tensorboard

`test.py` `test_ensemble.py` ensemble to predict

`model.py` `model_test.py` custom CNN model

`dataset.py` image augmentation & custom dataset

`util.py` save & load model

</br>

## 4. 트러블 슈팅
### 정확도 부족 문제
- 가중치 초기화, augmentation, resnet 모델에서 핵심 기술인 residual block을 구현하였다.
