import os
import cv2
import json
import torch
import logging
import detectron2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from dcfs.dataloader import build_detection_test_loader
from detectron2.data import MetadataCatalog
from .archs import clip
import copy

logger = logging.getLogger(__name__)

class SimpleMLP(nn.Module):
    """
    轻量化多层感知机：小权重初始化+缩小隐藏层规模，避免主导特征变换
    """
    def __init__(self, in_features, out_features, hidden_ratio=3, act_layer=nn.GELU, drop_rate=0.05):
        super().__init__()
        # 缩小隐藏层维度（hidden_ratio从2→3，隐藏层更小）
        hidden_features = max(in_features // hidden_ratio, out_features)  # 确保隐藏层不小于输出维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        # -------------------------- 关键优化：小权重初始化 --------------------------
        # fc1/fc2用Xavier初始化并缩放（0.5倍），避免初始权重过大
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.constant_(self.fc1.bias, 0.0)  # 偏置初始化为0，减少偏移
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)  # GELU激活更平缓，避免极端值
        x = self.drop(x)
        x = self.fc2(x)
        return x


class CrossSelfAttentionManual(nn.Module):
    """
    优化核心：残差缩放+弱正则化+小参数初始化，确保新增模块不主导特征输出
    """
    def __init__(self, dim=768, num_heads=4, drop_rate=0.05, ffn_hidden_ratio=3, residual_scale=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 注意力缩放因子（保留原逻辑）
        self.residual_scale = residual_scale  # FFN输出的残差缩放系数（核心控制参数）

        # ============ Self-Attention 模块（温和化改造） ============
        # LayerNorm：调大epsilon减少数值波动，避免过度归一化
        self.norm_self_attn = nn.LayerNorm(dim, eps=1e-6)
        self.q_self = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.proj_self = nn.Linear(dim, dim)  # 自注意力输出投影
        self.drop_self_attn = nn.Dropout(drop_rate)  # 降低Dropout速率

        # Self-Attention后的FFN（小权重+残差缩放）
        self.norm_self_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_self = SimpleMLP(dim, dim, hidden_ratio=ffn_hidden_ratio, drop_rate=drop_rate)

        # ============ Cross-Attention 模块（同逻辑温和化） ============
        self.norm_cross_attn = nn.LayerNorm(dim, eps=1e-6)
        self.q_cross = nn.Linear(dim, dim)
        self.k_cross = nn.Linear(dim, dim)
        self.v_cross = nn.Linear(dim, dim)
        self.proj_cross = nn.Linear(dim, dim)  # 交叉注意力输出投影
        self.drop_cross_attn = nn.Dropout(drop_rate)

        # Cross-Attention后的FFN（小权重+残差缩放）
        self.norm_cross_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_cross = SimpleMLP(dim, dim, hidden_ratio=ffn_hidden_ratio, drop_rate=drop_rate)

        # 最终融合的Dropout（保留原逻辑，速率同步降低）
        self.drop = nn.Dropout(drop_rate)

        # -------------------------- 关键优化：注意力层参数初始化 --------------------------
        # 所有Q/K/V/投影层用小权重初始化，避免初始特征偏移
        for module in [self.q_self, self.k_self, self.v_self, self.proj_self,
                       self.q_cross, self.k_cross, self.v_cross, self.proj_cross]:
            nn.init.xavier_uniform_(module.weight, gain=0.3)  # 增益0.3（更小权重）
            nn.init.constant_(module.bias, 0.0)  # 偏置置0

    def forward(self, img_feat, text_feat):
        """
        img_feat: (N, dim) - 图像特征（核心特征，保留主导地位）
        text_feat: (C, dim) - 文本特征（辅助特征，不主导）
        """
        N, _ = img_feat.shape
        C, _ = text_feat.shape
        H = self.num_heads
        
        # ----------------- 1. Self-Attention（弱干预逻辑） -----------------
        # Step1: LayerNorm（温和归一化）→ 注意力计算（保留原逻辑）
        x_sa = self.norm_self_attn(img_feat)  # 仅对输入做轻微归一化，不改变整体分布
        q = self.q_self(x_sa).view(N, H, self.head_dim).transpose(0, 1)
        k = self.k_self(x_sa).view(N, H, self.head_dim).transpose(0, 1)
        v = self.v_self(x_sa).view(N, H, self.head_dim).transpose(0, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # 移除原代码中不必要的 "/2"，保持标准注意力逻辑
        
        # Step2: 投影→Dropout→残差连接（无缩放，保留自注意力核心结果）
        sa_output = (attn @ v).transpose(0, 1).contiguous().view(N, self.dim)
        sa_output = self.proj_self(sa_output)
        sa_output = self.drop_self_attn(sa_output)
        sa_intermediate = img_feat + sa_output  # 自注意力结果直接残差，确保其作用

        # Step3: FFN（弱干预）→ 残差缩放（核心控制：FFN输出仅占10%）
        sa_ffn_output = self.norm_self_ffn(sa_intermediate)
        sa_ffn_output = self.ffn_self(sa_ffn_output)  # 小权重FFN，输出幅度小
        sa_out = sa_intermediate + self.residual_scale * sa_ffn_output  # FFN贡献仅10%

        # ----------------- 2. Cross-Attention（同弱干预逻辑） -----------------
        # Step1: LayerNorm→注意力计算（文本特征不做Norm，避免破坏预训练分布）
        x_ca = self.norm_cross_attn(sa_out)
        q_c = self.q_cross(x_ca).view(N, H, self.head_dim).transpose(0, 1)
        k_c = self.k_cross(text_feat).view(C, H, self.head_dim).transpose(0, 1)  # 文本特征直接用
        v_c = self.v_cross(text_feat).view(C, H, self.head_dim).transpose(0, 1)

        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.scale
        attn_c = F.softmax(attn_c, dim=-1)  # 移除原代码中不必要的 "/2"
        
        # Step2: 投影→Dropout→残差连接（无缩放，保留交叉注意力核心结果）
        ca_output = (attn_c @ v_c).transpose(0, 1).contiguous().view(N, self.dim)
        ca_output = self.proj_cross(ca_output)
        ca_output = self.drop_cross_attn(ca_output)
        ca_intermediate = sa_out + ca_output  # 交叉注意力结果直接残差

        # Step3: FFN（弱干预）→ 残差缩放（FFN贡献仅10%）
        ca_ffn_output = self.norm_cross_ffn(ca_intermediate)
        ca_ffn_output = self.ffn_cross(ca_ffn_output)
        ca_out = ca_intermediate + self.residual_scale * ca_ffn_output  # 控制FFN影响

        # ----------------- 3. 最终融合（严格保留原逻辑主导） -----------------
        # 交叉注意力结果（ca_out）仅占10%，原始图像特征（img_feat）占90%，确保核心逻辑不变
        fused = img_feat + self.drop(0.05 * ca_out)
        fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)  # 保留原L2归一化
        
        return fused


class CMCLIP:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.CMCLIP_ALPHA
        # -------------------------- 关键优化：实例化时传入温和参数 --------------------------
        self.cross_self_attn = CrossSelfAttentionManual(
            dim=768,
            num_heads=4,          # 保留原头数，不改变注意力粒度
            drop_rate=0.05,       # 降低Dropout，减少信息丢失
            ffn_hidden_ratio=3,   # 缩小FFN规模，减弱改造强度
            residual_scale=0.1    # FFN残差缩放，仅贡献10%
        ).to(self.device)
        self.cross_self_attn.eval()  # 确保测试时Dropout关闭

        # 以下部分完全保留原逻辑，避免其他模块引入干扰
        self.imagenet_model, self.preprocess = clip.load("ViT-L/14@336px")
        self.imagenet_model.cuda().eval()
        self.input_resolution = self.imagenet_model.visual.input_resolution
        self.context_length = self.imagenet_model.context_length
        self.vocab_size = self.imagenet_model.vocab_size
        
        self.dataloader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0])
        self.roi_pooler = ROIPooler(
            output_size=(336, 336), 
            scales=(1,), 
            sampling_ratio=0, 
            pooler_type="ROIAlignV2"
        )
        
        self.exclude_cls = self.clsid_filter()
        self.class_vector = self.text_encode()

    # 以下方法（text_encode、extract_roi_features、execute_calibration、clsid_filter）完全保留原代码
    def text_encode(self):
        dsname = self.cfg.DATASETS.TEST[0]
        
        self.classes = copy.deepcopy(MetadataCatalog.get(dsname).thing_classes)
        novel_id = copy.deepcopy(MetadataCatalog.get(dsname).get("novel_dataset_id_to_contiguous_id"))
        if novel_id is not None:
            thing_id = copy.deepcopy(MetadataCatalog.get(dsname).thing_dataset_id_to_contiguous_id)
            self.class_mapper = {thing_id[k]: idx for idx, k in enumerate(novel_id.keys())}
        elif 'voc' in dsname:
            self.class_mapper = {k: idx for idx, k in enumerate(range(15, 20))}
        else:
            print('implement class mapper!')
            raise NotImplementedError
        
        prompts = []
        self.classes.append('background')
        for idx, _class in enumerate(self.classes):
            if idx in self.exclude_cls:
                continue
            prompt = f"a photo of {_class}"
            prompts.append(prompt)
        
        text_tokens = clip.tokenize(prompts).cuda()
        with torch.no_grad():
            text_features = self.imagenet_model.encode_text(text_tokens)
        text_features = text_features.to(torch.float32)    
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def extract_roi_features(self, img, boxes):
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)

        images = [img / 255.]
        images = ImageList.from_tensors(images, 0)

        box_features = self.roi_pooler([images.tensor], boxes)

        if len(box_features) == 0:
            return torch.zeros(1, 768, dtype=torch.float32).to(self.device), []

        with torch.no_grad():
            conv_feature = self.imagenet_model.encode_image(box_features)
        conv_feature = conv_feature.to(torch.float32)
        return conv_feature, box_features

    def execute_calibration(self, inputs, dts):
        img = cv2.imread(inputs[0]['file_name'])
        img_id = inputs[0]['image_id']

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.CMCLIP_UPPER).sum().detach().cpu().numpy()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.CMCLIP_LOWER).sum().detach().cpu().numpy()
        assert ileft <= iright

        idx = []
        pred_class_list = []
        for i in range(ileft, iright):
            pred_class = int(dts[0]['instances'].pred_classes[i])
            if pred_class in self.exclude_cls:
                continue
            if pred_class not in self.class_mapper:
                continue
            pred_class_list.append(self.class_mapper[pred_class])
            idx.append(i)

        if not idx:
            return dts

        idx = np.array(idx)
        pred_class_list = np.array(pred_class_list)
        boxes = [dts[0]['instances'].pred_boxes[idx]]

        features, box_features = self.extract_roi_features(img, boxes)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)

        dsname = self.cfg.DATASETS.TEST[0]
        use_attention = not ('voc' in dsname.lower())

        if use_attention:
            with torch.no_grad():
                fused_features = self.cross_self_attn(features, self.class_vector)
        else:
            fused_features = features

        similarity = fused_features @ self.class_vector.T

        alpha = F.relu(similarity) + 1e-3
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        dirichlet_mean = alpha / alpha_sum

        softmax_score = torch.exp(similarity.float() * 100)
        softmax_score = softmax_score / torch.sum(softmax_score, dim=1, keepdim=True)

        score = 0.95 * softmax_score + 0.05 * dirichlet_mean

        dts[0]['instances'].scores[idx] = (
            dts[0]['instances'].scores[idx] * self.alpha +
            score[range(len(idx)), pred_class_list] * (1 - self.alpha)
        )

        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45,
                               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64, 65,
                               66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output