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
    
class CosineTripletLossWithL1(nn.Module):
    def __init__(self, margin=0.2, reg_weight=1.0, triplet_weight=1.0) -> None:
        super().__init__()
        self.triplet_loss = CosineTripletLoss(margin=margin)
        self.reg_loss = nn.L1Loss()
        self.reg_weight = reg_weight
        self.triplet_weight = triplet_weight
    
    def forward(self, anchor_feat, anchor_score, anchor_label, pos_feat, pos_score, pos_label, neg_feat, neg_score, neg_label):
        anchor_reg = self.reg_loss(anchor_score, anchor_label)
        pos_reg = self.reg_loss(pos_score, pos_label)
        neg_reg = self.reg_loss(neg_score, neg_label)
        triplet = self.triplet_loss(anchor_feat, pos_feat, neg_feat)
        return (anchor_reg + pos_reg + neg_reg) * self.reg_weight + triplet * self.triplet_weight, anchor_reg, pos_reg, neg_reg, triplet
        
if __name__ == "__main__":
    # 示例使用
    triplet_loss = CosineTripletLoss(margin=0.2)
    anchor = torch.randn(10, 128)  # 假设每个样本是128维
    positive = torch.randn(10, 128)
    negative = torch.randn(10, 128)

    loss = triplet_loss(anchor, positive, negative)
    print(loss)
