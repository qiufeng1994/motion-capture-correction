import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target.float())

def softmax_ce(output, target):
    return F.cross_entropy(output, target.long())

def sigmoid_ce(output, target):
    return F.binary_cross_entropy(output, target)

def softmax_cross_entropy_with_logits(output, target):
    loss = torch.sum(- target * F.log_softmax(output, -1), -1)
    mean_loss = loss.mean()
    return mean_loss

def bce(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def skeleton_loss(output, target):
    # mseloss
    mseloss = F.mse_loss(output, target[:,:,2,:].squeeze())
    # smoothloss
    neighbor_frames = target.permute(2,0,1,3).squeeze()
    neighbor_frames = [neighbor_frames[i] for i in range(len(neighbor_frames)) if i!= len(neighbor_frames)//2 ]
    smoothloss = sum([F.mse_loss(output,t) for t in neighbor_frames])
    return mseloss + smoothloss * 0.5 # mseloss: 0.46, smoothloss: 1.88
# def bce(output, target):
#     return torch.nn.BCEWithLogitsLoss(output, target)