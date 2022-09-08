import torch.nn as nn
import torch

class GIN(nn.Module):
    def __init__(self):
        super(GIN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,2,3,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2, 3, 3, padding=1),
        )

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def forward(self, x):
        self.__initialize_weights()
        out1 = self.net(x)
        self.__initialize_weights()
        out2 = self.net(x)
        alpha = torch.rand(2).cuda()

        out1 = out1*alpha[0] + (1-alpha[0])*x
        out2 = out2*alpha[1] + (1-alpha[1])*x

        out1 = out1 * ((torch.square(x).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True)).sqrt()/
                       (torch.square(out1).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True)).sqrt())

        out2 = out2 * ((torch.square(x).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True)).sqrt() /
                       (torch.square(out2).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True).sum(dim=1,keepdim=True)).sqrt())
        return out1, out2