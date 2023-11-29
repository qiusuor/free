import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算余弦相似度
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        # 计算anchor和positive之间的余弦相似度
        positive_similarity = cos_similarity(anchor, positive)

        # 计算anchor和negative之间的余弦相似度
        negative_similarity = cos_similarity(anchor, negative)

        # 计算损失
        loss = F.relu(self.margin + positive_similarity - negative_similarity)

        return loss.mean()

if __name__ == "__main__":
    # 示例使用
    triplet_loss = CosineTripletLoss(margin=0.2)
    anchor = torch.randn(10, 128)  # 假设每个样本是128维
    positive = torch.randn(10, 128)
    negative = torch.randn(10, 128)

    loss = triplet_loss(anchor, positive, negative)
    print(loss)
