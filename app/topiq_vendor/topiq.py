"""TOP-IQ metric, proposed by

TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment.
Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin.
Transactions on Image Processing, 2024.

Paper link: https://arxiv.org/abs/2308.03060

"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import timm
import copy
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def dist_to_mos(dist_score):
    num_classes = dist_score.shape[-1]
    return (dist_score * torch.arange(1, num_classes + 1).to(dist_score)).sum(dim=-1, keepdim=True)


def _get_clones(module, N):
    """Return ``N`` deep-copied modules in a :class:`~torch.nn.ModuleList`."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Resolve activation callable from string name.

    Args:
        activation (str): Activation name. Supported values are ``'relu'``,
            ``'gelu'``, and ``'glu'``.

    Returns:
        Callable: Activation function from :mod:`torch.nn.functional`.

    Raises:
        RuntimeError: If ``activation`` is unsupported.
    """
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer used in local self-attention blocks."""
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, src):
        src2 = self.norm1(src)
        q = k = src2
        src2, self.attn_map = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer used for cross-scale attention."""
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation='gelu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory):
        memory = self.norm2(memory)
        tgt2 = self.norm1(tgt)
        tgt2, self.attn_map = self.multihead_attn(query=tgt2, key=memory, value=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):
    """Stacked wrapper for encoder layers."""
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src

        for layer in self.layers:
            output = layer(output)

        return output


class TransformerDecoder(nn.Module):
    """Stacked wrapper for decoder layers."""
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory)

        return output


class GatedConv(nn.Module):
    """Gated local pooling module for no-reference feature aggregation."""
    def __init__(self, weightdim, ksz=3):
        super().__init__()

        self.splitconv = nn.Conv2d(weightdim, weightdim * 2, 1, 1, 0)
        self.act = nn.GELU()

        self.weight_blk = nn.Sequential(
            nn.Conv2d(weightdim, 64, 1, stride=1),
            nn.GELU(),
            nn.Conv2d(64, 64, ksz, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, ksz, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1, x2 = self.splitconv(x).chunk(2, dim=1)
        weight = self.weight_blk(x2)
        x1 = self.act(x1)
        return x1 * weight


class CFANet(nn.Module):
    """TOPIQ/CFANet architecture for NR and FR quality prediction.

    Args:
        semantic_model_name (str): Backbone name, for example ``'resnet50'``,
            ``'clip_ViT-B/32'``, or a Swin variant.
        model_name (str): Registered checkpoint key.
        backbone_pretrain (bool): Whether to load pretrained backbone weights.
        in_size (tuple[int, int] | None): Optional training input size.
        use_ref (bool): Whether to use a reference image input.
        num_class (int): Number of output dimensions.
        num_crop (int): Number of evaluation crops.
        crop_size (int): Crop size for multi-crop evaluation.
        inter_dim (int): Intermediate feature dimension.
        num_heads (int): Attention head count.
        num_attn_layers (int): Number of attention layers per block.
        dprate (float): Dropout probability.
        activation (str): Activation name.
        pretrained (bool): Whether to load pretrained CFANet checkpoint.
        pretrained_model_path (str | None): Optional local checkpoint path.
        out_act (bool): Whether to apply positive output activation for scalar
            prediction.
        block_pool (str): Feature block pooling mode.
        test_img_size (tuple[int, int] | None): Optional test-time resize.
        align_crop_face (bool): Whether to run face alignment for GFIQA models.
        default_mean (tuple[float, float, float]): Input normalization mean.
        default_std (tuple[float, float, float]): Input normalization std.

    Notes:
        Set ``use_ref=True`` for full-reference mode and ``use_ref=False`` for
        no-reference mode.
    """
    def __init__(
        self,
        semantic_model_name='resnet50',
        model_name='cfanet_nr_koniq_res50',
        backbone_pretrain=True,
        in_size=None,
        use_ref=True,
        num_class=1,
        num_crop=1,
        crop_size=256,
        inter_dim=256,
        num_heads=4,
        num_attn_layers=1,
        dprate=0.1,
        activation='gelu',
        pretrained=True,
        pretrained_model_path=None,
        out_act=False,
        block_pool='weighted_avg',
        test_img_size=None,
        align_crop_face=True,
        default_mean=IMAGENET_DEFAULT_MEAN,
        default_std=IMAGENET_DEFAULT_STD,
    ):
        super().__init__()

        self.in_size = in_size

        self.model_name = model_name
        self.semantic_model_name = semantic_model_name
        self.semantic_level = -1
        self.crop_size = crop_size
        self.use_ref = use_ref

        self.num_class = num_class
        self.block_pool = block_pool
        self.test_img_size = test_img_size

        self.align_crop_face = align_crop_face

        # =============================================================
        # define semantic backbone network
        # =============================================================

        if (semantic_model_name != 'resnet50' or model_name != 'cfanet_iaa_ava_res50'
                or pretrained or backbone_pretrain or pretrained_model_path is not None
                or use_ref or num_crop != 1 or num_class != 10):
            raise ValueError('Vendored TOPIQ supports only offline AVA ResNet50 single-image inference')
        self.semantic_model = timm.create_model('resnet50', pretrained=False, features_only=True)
        feature_dim_list = self.semantic_model.feature_info.channels()
        self.fix_bn(self.semantic_model)

        self.default_mean = torch.Tensor(default_mean).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(default_std).view(1, 3, 1, 1)

        # =============================================================
        # define self-attention and cross scale attention blocks
        # =============================================================

        self.fusion_mul = 3 if use_ref else 1
        ca_layers = sa_layers = num_attn_layers

        self.act_layer = nn.GELU() if activation == 'gelu' else nn.ReLU()
        dim_feedforward = min(4 * inter_dim, 2048)

        # gated local pooling and self-attention
        tmp_layer = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )
        self.sa_attn_blks = nn.ModuleList()
        self.dim_reduce = nn.ModuleList()
        self.weight_pool = nn.ModuleList()
        for idx, dim in enumerate(feature_dim_list):
            dim = dim * 3 if use_ref else dim
            if use_ref:
                self.weight_pool.append(
                    nn.Sequential(
                        nn.Conv2d(dim // 3, 64, 1, stride=1),
                        self.act_layer,
                        nn.Conv2d(64, 64, 3, stride=1, padding=1),
                        self.act_layer,
                        nn.Conv2d(64, 1, 3, stride=1, padding=1),
                        nn.Sigmoid(),
                    )
                )
            else:
                self.weight_pool.append(GatedConv(dim))

            self.dim_reduce.append(
                nn.Sequential(
                    nn.Conv2d(dim, inter_dim, 1, 1),
                    self.act_layer,
                )
            )

            self.sa_attn_blks.append(TransformerEncoder(tmp_layer, sa_layers))

        # cross scale attention
        self.attn_blks = nn.ModuleList()
        tmp_layer = TransformerDecoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )
        for i in range(len(feature_dim_list) - 1):
            self.attn_blks.append(TransformerDecoder(tmp_layer, ca_layers))

        # attention pooling and MLP layers
        self.attn_pool = TransformerEncoderLayer(
            inter_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            normalize_before=True,
            dropout=dprate,
            activation=activation,
        )

        linear_dim = inter_dim
        self.score_linear = [
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.LayerNorm(linear_dim),
            nn.Linear(linear_dim, linear_dim),
            self.act_layer,
            nn.Linear(linear_dim, self.num_class),
        ]

        # make sure output is positive, useful for 2AFC datasets with probability labels
        if out_act and self.num_class == 1:
            self.score_linear.append(nn.Softplus())

        if self.num_class > 1:
            self.score_linear.append(nn.Softmax(dim=-1))

        self.score_linear = nn.Sequential(*self.score_linear)

        self.h_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 32, 1))
        self.w_emb = nn.Parameter(torch.randn(1, inter_dim // 2, 1, 32))

        nn.init.trunc_normal_(self.h_emb.data, std=0.02)
        nn.init.trunc_normal_(self.w_emb.data, std=0.02)
        self._init_linear(self.dim_reduce)
        self._init_linear(self.sa_attn_blks)
        self._init_linear(self.attn_blks)
        self._init_linear(self.attn_pool)

        self.eps = 1e-8
        self.crops = num_crop

    def _init_linear(self, m):
        for module in m.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                nn.init.constant_(module.bias.data, 0)

    def preprocess(self, x):
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)
        return x

    def fix_bn(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

    def dist_func(self, x, y, eps=1e-12):
        return torch.sqrt((x - y) ** 2 + eps)

    def forward_cross_attention(self, x, y=None):
        # resize image when testing
        if not self.training:
            if self.test_img_size is not None:
                x = TF.resize(x, self.test_img_size, antialias=True)

        x = self.preprocess(x)
        if self.use_ref:
            y = self.preprocess(y)

        dist_feat_list = self.semantic_model(x)
        self.fix_bn(self.semantic_model)
        self.semantic_model.eval()

        start_level = 0
        end_level = len(dist_feat_list)

        b, c, th, tw = dist_feat_list[end_level - 1].shape
        pos_emb = torch.cat(
            (
                self.h_emb.repeat(1, 1, 1, self.w_emb.shape[3]),
                self.w_emb.repeat(1, 1, self.h_emb.shape[2], 1),
            ),
            dim=1,
        )

        token_feat_list = []
        for i in reversed(range(start_level, end_level)):
            tmp_dist_feat = dist_feat_list[i]

            # gated local pooling
            if self.use_ref:
                tmp_ref_feat = ref_feat_list[i]
                diff = self.dist_func(tmp_dist_feat, tmp_ref_feat)

                tmp_feat = torch.cat([tmp_dist_feat, tmp_ref_feat, diff], dim=1)
                weight = self.weight_pool[i](diff)
                tmp_feat = tmp_feat * weight
            else:
                tmp_feat = self.weight_pool[i](tmp_dist_feat)

            if tmp_feat.shape[2] > th and tmp_feat.shape[3] > tw:
                tmp_feat = F.adaptive_avg_pool2d(tmp_feat, (th, tw))

            # self attention
            tmp_pos_emb = F.interpolate(
                pos_emb, size=tmp_feat.shape[2:], mode='bicubic', align_corners=False
            )
            tmp_pos_emb = tmp_pos_emb.flatten(2).permute(2, 0, 1)

            tmp_feat = self.dim_reduce[i](tmp_feat)
            tmp_feat = tmp_feat.flatten(2).permute(2, 0, 1)
            tmp_feat = tmp_feat + tmp_pos_emb

            tmp_feat = self.sa_attn_blks[i](tmp_feat)
            token_feat_list.append(tmp_feat)

        # high level -> low level: coarse to fine
        query = token_feat_list[0]
        query_list = [query]
        for i in range(len(token_feat_list) - 1):
            key_value = token_feat_list[i + 1]
            query = self.attn_blks[i](query, key_value)
            query_list.append(query)

        final_feat = self.attn_pool(query)
        out_score = self.score_linear(final_feat.mean(dim=0))

        return out_score

    def forward(self, x):
        return dist_to_mos(self.forward_cross_attention(x))
