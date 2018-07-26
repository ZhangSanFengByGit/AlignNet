import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def kron_matching(*inputs):
    assert len(inputs) == 2
    assert inputs[0].dim() == 4 and inputs[1].dim() == 4
    assert inputs[0].size() == inputs[1].size()
    N, C, H, W = inputs[0].size()

    # Convolve every feature vector from inputs[0] with inputs[1]
    #   In: x0, x1 = N x C x H x W
    #   Proc: weight = x0, permute to (NxHxW) x C x 1 x 1
    #         input = x1, view as 1 x (NxC) x H x W
    #   Out: out = F.conv2d(input, weight, groups=N)
    #            = 1 x (NxHxW) x H x W, view as N x H x W x (HxW)
    w = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, C, 1, 1)
    x = inputs[1].view(1, N * C, H, W)
    x = F.conv2d(x, w, groups=N)
    x = x.view(N, H, W, H, W)
    return x


class KronMatching(nn.Module):
    def __init__(self):
        super(KronMatching, self).__init__()

    def forward(self, *inputs):
        return kron_matching(*inputs)


class KronEmbed(nn.Module):
    #def __init__(self, num_features=0, num_classes=0, dropout=0.5):
    def __init__(self, num_features=0):
        super(KronEmbed, self).__init__()
        self.kron = KronMatching()
        self.bn = nn.BatchNorm1d(num_features)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        #self.classifier = nn.Linear(num_features, num_classes)
        #self.classifier.weight.data.normal_(0, 0.001)
        #self.classifier.bias.data.zero_()
        #self.drop = nn.Dropout(dropout)

    def _kron_matching(self, x1, x2):
        if not self.training and len(x1.size()) != len(x2.size()):
            if len(x2.size()) < len(x1.size()):
                x2 = x2.unsqueeze(0)
                x2 = x2.expand_as(x1)
                x2 = x2.contiguous()
            if len(x2.size()) > len(x1.size()):
                x1 = x1.unsqueeze(0)
                x1 = x1.expand_as(x2)
                x1 = x1.contiguous()
        n = x1.size(0)
        c = x1.size(1)
        h = x1.size(2)
        w = x1.size(3)
        x2_kro = self.kron(x1 / x1.norm(2, 1, keepdim=True).expand_as(x1),
                           x2 / x2.norm(2, 1, keepdim=True).expand_as(x2))
        x2_kro_att = F.softmax((1.0 * x2_kro).view(n * h * w, h * w), dim=1).view(n, h, w, h, w)
        masked_x2 = torch.bmm(x2.view(n, c, h * w), x2_kro_att.view(n, h * w, h * w).transpose(1, 2)).view(n, c, h, w)
        return masked_x2
        #x = masked_x2 - x1
        #x = F.avg_pool2d(x, x.size()[2:])
        #x = x.view(n, -1)
        #return x

    def forward(self, x1, x2):
        # Conduct Kron Match between feature maps
        probe_x = x1
        gallery_x = x2
        x = self._kron_matching(probe_x, gallery_x)
        #x = x.pow(2)
        #x = self.bn(x)
        #x = self.drop(x)
        #x = self.classifier(x)
        return x





        