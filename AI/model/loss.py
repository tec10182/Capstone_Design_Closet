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
                    [features[: start_idx + i], features[start_idx + i + 1 :]]
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
    def __init__(self, patience=15, min_delta=0):
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
    total_outgroup_diff = 0.0  # 그룹 외 벡터들과의 유사도 계산 추가
    num_groups = len(num_crops)

    n = 0
    for g, num_crop in enumerate(num_crops):
        # Positive features group
        pos_features = features[start_idx : start_idx + num_crop]

        # Compute average cosine similarity for the positive pairs
        similarities = []
        outgroup_similarities = []

        # Calculate pairwise cosine similarity within the group
        for i in range(num_crop):
            for j in range(num_crop):
                if i != j:
                    sim = F.cosine_similarity(
                        pos_features[i].unsqueeze(0),
                        pos_features[j].unsqueeze(0),
                        dim=-1,
                    )
                    similarities.append(sim.item())

            # Calculate cosine similarity with features from other groups
            outgroup_features = torch.cat(
                [features[:start_idx], features[start_idx + num_crop :]], dim=0
            )
            for out_feat in outgroup_features:
                out_sim = F.cosine_similarity(pos_features[i], out_feat, dim=-1)
                outgroup_similarities.append(out_sim.item())

        # Average similarity within the group
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        total_diff += avg_similarity

        # Average similarity with out-group vectors
        avg_outgroup_similarity = (
            sum(outgroup_similarities) / len(outgroup_similarities)
            if outgroup_similarities
            else 0.0
        )
        total_outgroup_diff += avg_outgroup_similarity

        n += 1
        start_idx += num_crop

    # Return the average similarities within and outside the group
    return [total_diff, total_outgroup_diff, n]
