import random


# class RandomSquareCrop:
#     def __init__(self, min_size=30, r_range=(0.1, 0.25)):
#         self.min_size = min_size
#         self.r_range = r_range

#     def __call__(self, img):
#         w, h = img.size
#         r = random.uniform(self.r_range[0], self.r_range[1])
#         crop_size = min(self.min_size, int(r * min(w, h)))

#         # 랜덤으로 크롭의 왼쪽 상단 코너 선택
#         x = random.randint(0, w - crop_size)
#         y = random.randint(0, h - crop_size)

#         return img.crop((x, y, x + crop_size, y + crop_size))


class RandomSquareCrop:
    def __init__(self, min_size=30, r_range=(0.1, 0.25), apply_prob=0.0):
        self.min_size = min_size
        self.r_range = r_range
        self.apply_prob = apply_prob

    def __call__(self, img):
        # 0.5 확률로 적용
        if random.random() > self.apply_prob:
            return img  # 확률을 충족하지 않으면 원본 이미지 반환

        w, h = img.size
        r = random.uniform(self.r_range[0], self.r_range[1])
        crop_size = min(self.min_size, int(r * min(w, h)))

        # 랜덤으로 크롭의 왼쪽 상단 코너 선택
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)

        return img.crop((x, y, x + crop_size, y + crop_size))
