# Capstone_Design_Closet

## 실행 코드
```
.\train.ps1
```


## train.ps1의 parameter 수정

| parameter   | 의미   | 값(default)   |
|------------|------------|------------|
| $BATCH_SIZE | 배치 크기 | 정수(32) |
|$GENERATOR | 사용할 CNN | microsoft/resnet-50|
|$IMAGE_DIR| 이미지가 저장된 경로 | "d:\dataset\project_dataset" |
|$TRAIN_DIR| train 데이터의 메타데이터 경로 | "d:\dataset\train.csv" |
|$VAL_DIR | val 데이터의 메타데이터 경로 | "d:\dataset\train.csv" |
|$SAVE_DIR | 모델 path를 저장할 경로 | "d:\model\path" |