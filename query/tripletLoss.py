import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None, device="cpu"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device = device
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1).to(device=self.device)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        positive_similarity = cos_similarity(anchor, positive)
        negative_similarity = cos_similarity(anchor, negative)

        loss = F.relu(self.margin - positive_similarity + negative_similarity)

        return loss.mean()
       
class TripletWrapLoss(nn.Module):
    def __init__(self, margin=0.5, loss=nn.L1Loss(), loss_weight=1.0, triplet_weight=10.0, device="cuda", method="cosine") -> None:
        super().__init__()
        self.triplet_loss = CosineTripletLoss(margin=margin) if method == "cosine" else TripletLoss(margin=margin, device=device)
        self.reg_loss = loss
        self.loss_weight = loss_weight
        self.triplet_weight = triplet_weight
    
    def forward(self, anchor_feat, anchor_score, anchor_label, pos_feat, pos_score, pos_label, neg_feat, neg_score, neg_label):
        anchor_reg = self.reg_loss(anchor_score, anchor_label)
        pos_reg = self.reg_loss(pos_score, pos_label)
        neg_reg = self.reg_loss(neg_score, neg_label)
        triplet = self.triplet_loss(anchor_feat, pos_feat, neg_feat)
        return (anchor_reg + pos_reg + neg_reg) * self.loss_weight + triplet * self.triplet_weight, anchor_reg, pos_reg, neg_reg, triplet
    
     
if __name__ == "__main__":
    # 示例使用
    triplet_loss = CosineTripletLoss(margin=0.05)
    anchor = torch.randn(10, 128)  # 假设每个样本是128维
    positive = torch.randn(10, 128)
    negative = torch.randn(10, 128)

    loss = triplet_loss(anchor, positive, negative)
    print(loss)
