import torch.nn as nn
import torch
from torch.autograd import Function
from torch.nn import init

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
        self.style_dim = style_dim

    def forward(self, x):
        s = torch.randn(x.shape[0],self.style_dim).cuda()
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class NormalNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            mu = weight_mat.mean()
            std = weight_mat.std()
            # print(mu,std)
        weight_sn = (weight-mu) / std

        return weight_sn

    @staticmethod
    def apply(module, name):
        fn = NormalNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        module.register_buffer(name, weight)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn = self.compute_weight(module)
        setattr(module, self.name, weight_sn)


def spectral_norm(module, name='weight'):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        NormalNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight, gain)

    return spectral_norm(module)

class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        ch = args.GIN_ch
        self.net1 = nn.Sequential(
            nn.Conv2d(3,ch,3,padding=1),
            (AdaIN(2,ch) if args.noise else nn.Identity()),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) if args.noise else nn.Identity()),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) if args.noise else nn.Identity()),
            nn.LeakyReLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            (AdaIN(2,ch) if args.noise else nn.Identity()),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) if args.noise else nn.Identity()),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            (AdaIN(2,ch) if args.noise else nn.Identity()),
            nn.LeakyReLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

        self.adv = args.adv
        self.__initialize_weights()
        if self.adv:
            self.apply(spectral_init)

        print(self.net1, self.net2)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                weight = m.weight.data*1.0
                m.weight.data = (m.weight.data - weight.mean())/weight.std()

    def forward(self, x):
        if not self.adv:
            self.__initialize_weights()

        out1 = self.net1(x)
        out2 = self.net2(x)

        return out1, out2