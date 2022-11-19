import random

import numpy as np
from torch import autograd

from networks.blocks import *
from networks.loss import *
from utils import batched_index_select, batched_scatter
from networks.vit import Attention

class LoFGAN_MHA2(nn.Module):
    def __init__(self, config):
        super(LoFGAN_MHA2, self).__init__()

        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])

        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']

    def forward(self, xs, y, mode):
        if mode == 'gen_update':
            fake_x, similarity, indices_feat, indices_ref, base_index = self.gen(xs)

            loss_recon = local_recon_criterion(xs, fake_x, similarity, indices_feat, indices_ref, base_index, s=8)

            feat_real, _, _ = self.dis(xs)
            feat_fake, logit_adv_fake, logit_c_fake = self.dis(fake_x)
            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

            loss_recon = loss_recon * self.w_recon      # Local Reconstruction Loss
            loss_adv_gen = loss_adv_gen * self.w_adv_g  # gen_Adversarial Loss
            loss_cls_gen = loss_cls_gen * self.w_cls    # gen_Classification Loss

            loss_total = loss_recon + loss_adv_gen + loss_cls_gen
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            xs.requires_grad_()

            _, logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d                # dis_Adversarial Loss for real
            loss_adv_dis_real.backward(retain_graph=True)

            #这里y就是label,logit_c_real[12,8]，其中12=batch_size*3，鉴别器的鉴别类有num_classes = 8
            y_extend = y.repeat(1, self.n_sample).view(-1)     #[0,0,0,6,6,6,1,1,1,3,3,3]
            index = torch.LongTensor(range(y_extend.size(0))).cuda()    #[0,1,2,3,4,5,6,7,8,9,10,11]
            logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)     #理应得到[12,1]
            loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)
            loss_reg_dis = loss_reg_dis * self.w_gp                 # dis_regression Loss(使用了GP)
            loss_reg_dis.backward(retain_graph=True)

            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls                # dis_Class Loss
            loss_cls_dis.backward()

            with torch.no_grad():
                fake_x = self.gen(xs)[0]

            _, logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d             #dis_Adversarial Loss for fake
            loss_adv_dis_fake.backward()

            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_reg_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x

    #计算梯度的平方
    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        nf = config['nf']
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']

        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]


        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)      #cnn_feature,对特征向量的卷积
        self.cnn_adv = nn.Sequential(*cnn_adv)  #cnn_Adversarial: Adversarial Loss.
        self.cnn_c = nn.Sequential(*cnn_c)      #cnn_class: Classification Loss.

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        feat = self.cnn_f(x)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        return feat, logit_adv, logit_c


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fusion = LocalFusionModule(inplanes=128, rate=config['rate'])

    def forward(self, xs):
        b, k, C, H, W = xs.size()   #xs即为输入
        xs = xs.view(-1, C, H, W)   #特别注意，此处将b,k结合成一个维度
        querys = self.encoder(xs)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)     #输入数据经过encoder得到的

        similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  #得到一个[b,k]，数值随机的tensor
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # [b,k]
        similarity = similarity_total / similarity_sum  # [b,k] 类似将similarity_total中dim=1的数据，做归一化，累加和等于1

        base_index = random.choice(range(k))

        base_feat = querys[:, base_index, :, :, :]  #得到base_feature
        feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys, base_index, similarity)

        fake_x = self.decoder(feat_gen)     #输出生成图片fake_x

        return fake_x, similarity, indices_feat, indices_ref, base_index


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        model = [Conv2dBlock(3, 32, 5, 1, 2,    #[4*3,3,64,64]->[4*3,32,64,64]
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(32, 64, 3, 2, 1,  # 本来cnn不改变尺寸，此处缩小是因为stride=2,[4*3,64,32,32]
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Attention(64, num_heads=8, qkv_bias=True,
                            attn_drop=0.1, proj_drop=0.1),
                 Conv2dBlock(64, 128, 3, 2, 1,  # [4*3,128,16,16]
                         norm='bn',
                         activation='lrelu',
                         pad_type='reflect'),
                 Attention(128, num_heads=8, qkv_bias=True,
                           attn_drop=0.1, proj_drop=0.1),
                 Conv2dBlock(128, 128, 3, 2, 1,     #[4*3,128,8,8]
                         norm='bn',
                         activation='lrelu',
                         pad_type='reflect'),
                 Conv2dBlock(128, 128, 3, 2, 1,     #[4*3,128,4,4]
                         norm='bn',
                         activation='lrelu',
                         pad_type='reflect')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        model = [nn.Upsample(scale_factor=2),
                 Conv2dBlock(128, 128, 3, 1, 1,     #[4,128,4,4]->[4,128,8,8]
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(128, 128, 3, 1, 1,     #[4,64,16,16]
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Attention(128, num_heads=8, qkv_bias=True,
                           attn_drop=0.1, proj_drop=0.1),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(128, 64, 3, 1, 1,      #[4,64,32,32]
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Attention(64, num_heads=8, qkv_bias=True,
                           attn_drop=0.1, proj_drop=0.1),
                 nn.Upsample(scale_factor=2),
                 Conv2dBlock(64, 32, 3, 1, 1,       #[4,32,64,64]
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect'),
                 Conv2dBlock(32, 3, 5, 1, 2,  # [4,3,64,64]
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    '''
    feat:base_feature[b,1,c,h,w]
    refs:query,encoder输出的特征向量[b,k,c,h,w]
    index:base_feature的index
    similarity:相似度向量[b,k]
    '''
    def forward(self, feat, refs, index, similarity):
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)     #去掉base_feature
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # [b,1*h*w,c]=(32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # [b,2*h*w,c]=(32*128*128)

        # local selection
        rate = self.rate
        num = int(rate * h * w) #num=12
        '''
        random.sample(range(h * w), num):从0到h*w中，随机选出num个数，得到tensor[num]
        再重复b次后，在第0维拼接，得到feat_indices[b,num]
        目的是需要选择的各local_representation的index
        '''
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
                                 dim=0).cuda()  # B*num

        feat = feat.view(b, c, -1)  # [b,c,1*h*w]=(32*128*64)
        #按照feat_indeces的索引，从base_feature中选取local_representation
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # [b,c,num]=(32*128*12)

        # local matching
        #w_feat与feat的区别在于，前者做过归一化F.normalize()
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):  # n = 2
            ref = refs[:, j, :, :]  # [b,c,h*w]=(32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # [b,num,c]*[b,c,h*w]=[b,num,h*w]=(32*12*64)，余弦相乘
            _, indice = torch.topk(fx, dim=2, k=1)  # 返回dim=2维中，最大的点的index[b,num,1]，取h*w中那些最大值的下标
            indice = indice.squeeze(0).squeeze(-1)  # [b,num]=(32*12)，此处便代表论文中的similarity map
            #indice为base与fers点乘后，最大的点的下标index
            #按照indice的下标，在ref中进行选择，得到ref中，h*w里num个对应点：[b,c,h*w]->[b,c,num]
            select = batched_index_select(ref, dim=2, index=indice)  # [b,c,num]=(32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        #将得到的index与具体点进行拼接
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # [b,2,num]=(32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # [b,2,c,num]=(32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # [b,1,c*num]=(32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # [b,2,c*num]=(32*2*(128*12))

        #以一定几率similarity(随机得到)，将feat_select与ref_selects叠加
        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # [b,c,num]=(32*128*12)

        #再将得到的融合特征feat_fused，在base_feature中进行置换
        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # [b,c,h,w]=(32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)


if __name__ == '__main__':
    config = {}
    model = Generator(config).cuda()
    x = torch.randn(32, 3, 3, 128, 128).cuda()
    y, sim = model(x)
    print(y.size())
