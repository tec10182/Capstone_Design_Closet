# 변수 설정
$BATCH_SIZE = 4
$DOMAIN_ADAPTATION = "n"
$GENERATOR = "microsoft/resnet-50"
# $CHECKPOINT = 
# $DIMENSION = 
$IMAGE_DIR = "d:\dataset\project_dataset"
$TRAIN_DIR = "d:\dataset\train.csv"
$SAVE_DIR = "d:\model\path"
# $VAL_DIR = "d:\dataset\val.csv"
$VAL_DIR = "d:\dataset\train.csv"

# Python 스크립트 실행
python train.py `
    --batch_size $BATCH_SIZE `
    --domain_adaptation $DOMAIN_ADAPTATION `
    --generator $GENERATOR `
    --image_dir $IMAGE_DIR `
    --train_dir $TRAIN_DIR `
    --save $SAVE_DIR`
    --val_dir $VAL_DIR
