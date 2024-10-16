from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models
from model.embedder import Embedder


# Custom Dataset to load images directly from a folder without subfolders
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith(("jpg", "png", "jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert(
            "RGB"
        )  # Convert to RGB in case of grayscale
        if self.transform:
            image = self.transform(image)
        return image


# 이미지 전처리
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 이미지를 256x256으로 리사이즈
        transforms.ToTensor(),  # 이미지를 텐서로 변환
    ]
)

# Custom dataset을 이용하여 이미지 불러오기
image_dir = "./images/10"  # 이미지 폴더 경로
image_dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
image_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

# 모델 설정
generator = models.resnet50(pretrained=True)
generator.fc = torch.nn.Identity()  # FC 레이어 제거
embedder = Embedder()  # 임베더 모델

# 체크포인트 불러오기
checkpoint_path = r"d:\model\path\model_epoch_5.pt"
checkpoint = torch.load(checkpoint_path)
generator.load_state_dict(checkpoint["generator_state_dict"])
embedder.load_state_dict(checkpoint["embedder_state_dict"])

# 모델을 평가 모드로 전환
generator.eval()
embedder.eval()

# 이미지 처리 및 pair-wise cosine similarity 계산
for batch in tqdm(image_loader):
    with torch.no_grad():
        generator_output = generator(batch)  # generator로 이미지 처리
        embedded_output = embedder(generator_output)  # embedder로 임베딩 추출

    # Pair-wise cosine similarity 계산
    num_embeddings = embedded_output.size(0)

    similarities = []

    # 모든 pair-wise 코사인 유사도를 계산하여 저장
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):  # 자기 자신을 제외한 모든 pair를 계산
            cosine_sim = F.cosine_similarity(
                embedded_output[i], embedded_output[j], dim=0
            )
            similarities.append(cosine_sim.item())

    # Pair-wise cosine similarity의 평균 계산
    if similarities:
        print()
        avg_similarity = torch.mean(torch.tensor(similarities))  # 평균 계산
        print(f"Average Pair-wise Cosine Similarity: {avg_similarity:.4f}")
