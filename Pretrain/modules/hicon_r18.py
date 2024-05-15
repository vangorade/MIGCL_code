
import torch.nn as nn
import torchvision
from .resnet import *
from .resnet_imagenet import *

# Define the Vision Transformer Block
class ViTBlock(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(ViTBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# Define the Vision Transformer (ViT) model
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, out_dim, embed_dim):
        super(VisionTransformer, self).__init__()
        self.patch_embed = nn.Conv2d(512, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.vit_block = ViTBlock(in_channels=embed_dim, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=12)
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        # print(x.shape)
        cnn_features = self.patch_embed(x)

        vit_features = self.vit_block(cnn_features)
        vit_features = torch.cat([self.cls_token.expand(x.size(0), -1, -1), vit_features], dim=1)
        transformer_output = self.transformer(vit_features)
        cls_token_output = transformer_output[:, 0, :]
        output = self.fc(cls_token_output)

        return output

class HiCon_r18(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet to obtain hi = f(xi) = ResNet(xi) where hi is the output after the average pooling layer.
    """
    def __init__(self, args, data='non_imagenet'):
        super(HiCon_r18, self).__init__()
        self.args = args
        if data == 'imagenet':
            self.encoder = self.get_imagenet_resnet(args.resnet)
        else:
            self.encoder = self.get_resnet(args.resnet)

        self.n_features = self.encoder.feat_dim
        self.projector = nn.Sequential(nn.Linear(self.n_features, self.n_features),
                                       nn.ReLU(),
                                       nn.Linear(self.n_features, args.projection_dim))
        self.vit_model = VisionTransformer(
            image_size=224,  # Assuming input image size
            patch_size=4,    # Assuming a 16x16 patch size
            out_dim=args.projection_dim,
            embed_dim=768
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def get_resnet(self, name):
        resnets = {
            "resnet18": resnet18(data=self.args.dataset),
            "resnet34": resnet34(data=self.args.dataset),
            "resnet50": resnet50(data=self.args.dataset),
            "resnet101": resnet101(data=self.args.dataset),
            "resnet152": resnet152(data=self.args.dataset)}
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]
     
    def get_imagenet_resnet(self, name):
        resnets = {
            "resnet18": resnet18_imagenet(),
            "resnet34": resnet34_imagenet(),
            "resnet50": resnet50_imagenet(),
            "resnet101": resnet101_imagenet(),
            "resnet152": resnet152_imagenet()}
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]

    def forward(self, x):
        if self.args.model == 'LBE':
            mu2, mu3, mu4 = self.encoder(x)
            esp2 = mu2.data.new(mu2.size()).normal_(0., self.args.zeta)
            h2 = mu2 + esp2
            esp3 = mu3.data.new(mu3.size()).normal_(0., self.args.zeta)
            h3 = mu3 + esp3
            esp4 = mu4.data.new(mu4.size()).normal_(0., self.args.zeta)
            h4 = mu4 + esp4
            z = self.projector(h4)
            if self.args.normalize:
                z = nn.functional.normalize(z, dim=1)
            out = (mu2, mu3, mu4, h2, h3, h4, z)
        elif self.args.model == 'MIB':
            _, _, mu = self.encoder(x)
            esp = mu.data.new(mu.size()).normal_(0., self.args.zeta)
            h = mu + esp
            z = self.projector(h)
            if self.args.normalize:
                z = nn.functional.normalize(z, dim=1)
            out = (mu, h, z)
        else:
            _, _, h = self.encoder(x)
            h_pool = self.avgpool(h).squeeze()
            # print(h_pool.shape)
            z_cnn = self.projector(h_pool)
            # print(h.shape)
            z_trans = self.vit_model(h)
            # print(z_cnn.shape)
            # print(z_trans.shape)

            if self.args.normalize:
                z_cnn = nn.functional.normalize(z_cnn, dim=1)
                z_trans = nn.functional.normalize(z_trans, dim=1)

            # print(z_trans)
            out = (h_pool, z_cnn, z_trans)
        return out