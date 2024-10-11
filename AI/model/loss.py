import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


class ContrastiveLoss(nn.Module):
    def __init__(self, tau):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, features, num_crops):
        batch_size = len(features)
        total_loss = 0
        start_idx = 0

        # Loop through each group of crops defined by num_crops
        for num_crop in num_crops:
            # Positive features group
            pos_features = features[start_idx : start_idx + num_crop]

            # Anchor positive pairs (avoid i == j case)
            for i in range(num_crop):
                anchor = pos_features[i]

                # Positive similarities within the same group (excluding i == i)

                pos_sim = torch.exp(
                    F.cosine_similarity(anchor.unsqueeze(0), pos_features, dim=-1)
                    / self.tau
                )

                test = pos_sim
                pos_sim = pos_sim.sum() - torch.exp(
                    F.cosine_similarity(anchor, anchor, dim=0) / self.tau
                )

                # pos_sim = torch.clamp(
                #     pos_sim.sum()
                #     - torch.exp(F.cosine_similarity(anchor, anchor, dim=0) / self.tau),
                #     # min=1e-8,
                # )

                # Negative similarities from other groups
                neg_features = torch.cat(
                    [features[:start_idx], features[start_idx + num_crop :]]
                )
                neg_sim = torch.exp(
                    F.cosine_similarity(anchor.unsqueeze(0), neg_features, dim=-1)
                    / self.tau
                ).sum()

                loss = -torch.log(pos_sim / neg_sim)
                total_loss += loss

            start_idx += num_crop  # Move to the next group of crops
        return total_loss / batch_size


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): 개선되지 않아도 기다리는 에포크 수
            min_delta (float): 손실이 얼마나 개선되어야 하는지 최소 값
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True


def validate_similarity(features, num_crops):
    start_idx = 0
    total_diff = 0.0
    num_groups = len(num_crops)

    # Loop through each group of crops defined by num_crops
    n = 0
    for num_crop in num_crops:
        # Positive features group
        pos_features = features[start_idx : start_idx + num_crop]

        # Compute average cosine similarity for the positive pairs
        similarities = []

        # Calculate pairwise cosine similarity
        for i in range(num_crop):
            for j in range(num_crop):
                if i != j:  # 동일한 인덱스는 제외
                    sim = F.cosine_similarity(
                        pos_features[i].unsqueeze(0),
                        pos_features[j].unsqueeze(0),
                        dim=-1,
                    )
                    similarities.append(sim.item())  # pairwise similarity 추가
        avg_similarity = (
            sum(similarities) / len(similarities) if similarities else 0.0
        )  # Overall average for the group
        total_diff += avg_similarity
        n += 1
        start_idx += num_crop  # Move to the next group of crops
    # Return the average total_diff across all groups
    return [total_diff, n]
