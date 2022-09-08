from cut_model import ResnetGenerator, PatchSampleF, PatchNCELoss
import torch.nn as nn
import torch


crit  = PatchNCELoss().cuda()
def MI_loss(src, tgt, nets, nce_layers):
    # criterionNCE = []
    # for _ in nce_layers:
    #     criterionNCE.append(PatchNCELoss().cuda())

    netG, netF = nets

    n_layers = len(nce_layers)
    feat_q = netG(tgt, nce_layers, encode_only=True)

    feat_k = netG(src, nce_layers, encode_only=True)
    feat_k_pool, sample_ids = netF(feat_k, 256, None)
    feat_q_pool, _ = netF(feat_q, 256, sample_ids)

    bs = src.shape[0]

    total_nce_loss = 0.0
    for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, nce_layers):
        loss = crit(f_q, f_k, bs) * 1.0
        print(loss.mean())
        total_nce_loss += loss.mean()

    return total_nce_loss / n_layers

net = ResnetGenerator(input_nc=3, output_nc=3, ngf=8, norm_layer=nn.InstanceNorm2d,
                          use_dropout=False, no_antialias=False, no_antialias_up=False, n_blocks=6).cuda()

net_F = PatchSampleF(use_mlp=True, init_gain=0.02).cuda()

data1 = torch.randn(48, 3, 192, 192).cuda()

data2 = torch.randn(48, 3, 192, 192).cuda()

nce_layers = '0,4,8,12,16'
nce_layers = [int(i) for i in nce_layers.split(',')]

loss = MI_loss(data1, data2, [net, net_F], nce_layers)

print(loss)