# 변수 설정
$BATCH_SIZE = 2
$DOMAIN_ADAPTATION = "n"
$GENERATOR = "microsoft/resnet-50"
# $CHECKPOINT = 
# $DIMENSION = 
$IMAGE_DIR = "d:\dataset\project_dataset"
$META_DIR = "d:\dataset\label2.csv"
$SAVE_DIR = "d:\model\pathsh tr"

# Python 스크립트 실행
python train.py `
    --batch_size $BATCH_SIZE `
    --domain_adaptation $DOMAIN_ADAPTATION `
    --generator $GENERATOR `
    --image_dir $IMAGE_DIR `
    --meta_dir $META_DIR `
    --save $SAVE_DIR
