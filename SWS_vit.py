import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, default_cfgs, PatchEmbed

try:
    from timm.models.vision_transformer import HybridEmbed
except ImportError:
    # for higher version of timm
    from timm.models.vision_transformer_hybrid import HybridEmbed

import numpy as np



class RepeatedModuleList_norm(nn.Module):
    def __init__(self, Layer_num: int, sublist_repeated_times: list, \
                 sublist_with_flag: list, Sublist_instance, Default_instance=None, normalized_dim=None, *args, **kwargs):
        super().__init__()
        assert Layer_num == sum(sublist_repeated_times)
        assert len(sublist_repeated_times) == len(sublist_with_flag)

        self.Sublist_num = len(sublist_repeated_times)
        self.sublist_repeated_times = sublist_repeated_times
        self.sublist_with_flag = sublist_with_flag
        modules = []
        
        # self.stage_num, self.stage_count = np.unique(self.repeated_times_schedule['dim'], return_counts=True)

        assert (Default_instance is not None) or (False not in self.sublist_with_flag)

        i = 0
        for r, t in zip(self.sublist_repeated_times, self.sublist_with_flag):
            if (r==1 and Default_instance is not None) or not t:
                modules.append(Default_instance(normalized_shape = normalized_dim[i], *args, **kwargs))
            else:
                modules.append(Sublist_instance(normalized_shape = normalized_dim[i], *args, **kwargs))
            i = i + 1
        
        self.instances = nn.ModuleList(modules)

    def forward(self, *args, **kwargs):
        r = self._block_id
        return self.instances[r](*args, **kwargs)
    
    def layer_id_2_repeated_times(self):
        def get_block_id():
            s, i  = 0, 0
            while s <= self._layer_id:
                s += self.sublist_repeated_times[i]
                i+=1
            return i-1
        self._block_id = get_block_id()
        
        def set_repeated_id_fn(m):
            m._repeated_id = self._layer_id - sum(self.sublist_repeated_times[:self._block_id])
            m._ilayer = m._repeated_id/float(self.sublist_repeated_times[self._block_id])
            
        self.apply(set_repeated_id_fn)
        
    def __repr__(self):
        msg = super().__repr__()
        return msg


class RepeatedModuleList(nn.Module):
    def __init__(self, Layer_num: int, sublist_repeated_times: list, \
                 sublist_with_flag: list, Sublist_instance, Default_instance=None, in_dim=None, out_dim=None, *args, **kwargs):
        super().__init__()
        assert Layer_num == sum(sublist_repeated_times)
        assert len(sublist_repeated_times) == len(sublist_with_flag)

        self.Sublist_num = len(sublist_repeated_times)
        self.sublist_repeated_times = sublist_repeated_times
        self.sublist_with_flag = sublist_with_flag
        modules = []
        
        assert (Default_instance is not None) or (False not in self.sublist_with_flag)

        i = 0
        for r, t in zip(self.sublist_repeated_times, self.sublist_with_flag):
            if (r==1 and Default_instance is not None) or not t:
                modules.append(Default_instance(in_features=in_dim[i], out_features=out_dim[i], *args, **kwargs))
            else:
                modules.append(Sublist_instance(in_features=in_dim[i], out_features=out_dim[i], *args, **kwargs))
            i = i + 1

        self.instances = nn.ModuleList(modules)

    def forward(self, *args, **kwargs):
        r = self._block_id
        return self.instances[r](*args, **kwargs)
    
    def layer_id_2_repeated_times(self):
        def get_block_id():
            s, i  = 0, 0
            while s <= self._layer_id:
                s += self.sublist_repeated_times[i]
                i+=1
            return i-1
        self._block_id = get_block_id()
        
        def set_repeated_id_fn(m):
            m._repeated_id = self._layer_id - sum(self.sublist_repeated_times[:self._block_id])
            m._ilayer = m._repeated_id / float(self.sublist_repeated_times[self._block_id])
            
        self.apply(set_repeated_id_fn)
        
    def __repr__(self):
        msg = super().__repr__()
        return msg


class SWS_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., repeated_times_schedule=None, num_layers=None):
        super().__init__()
        assert isinstance(repeated_times_schedule, dict)
        self.repeated_times_schedule = repeated_times_schedule
        self.num_layers = num_layers
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        assert sum(self.repeated_times_schedule['mlp_fc1'][0]) == num_layers

        # self.repeated_times_dim = repeated_times_schedule['dim']

        self.fc1 = RepeatedModuleList(Layer_num=num_layers, sublist_repeated_times=self.repeated_times_schedule['mlp_fc1'][0], sublist_with_flag=self.repeated_times_schedule['mlp_fc1'][1], Sublist_instance=nn.Linear, Default_instance=nn.Linear, in_dim=in_features, out_dim=hidden_features, bias=bias[0])
            
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        
        assert sum(self.repeated_times_schedule['mlp_fc2'][0]) == num_layers

        self.fc2 = RepeatedModuleList(Layer_num=num_layers, sublist_repeated_times=self.repeated_times_schedule['mlp_fc2'][0], sublist_with_flag=self.repeated_times_schedule['mlp_fc2'][1], Sublist_instance=nn.Linear, Default_instance=nn.Linear, in_dim=hidden_features, out_dim=out_features, bias=bias[1])
        
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


    
    
class SWS_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., repeated_times_schedule=None, num_layers=None):
        super().__init__()
        assert isinstance(repeated_times_schedule, dict)
        self.repeated_times_schedule = repeated_times_schedule
        self.num_layers = num_layers
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.head_dim = head_dim = dim[0] // num_heads[0]

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.stage_num, self.stage_count = np.unique(self.repeated_times_schedule['dim'], return_counts=True)

        assert sum(self.repeated_times_schedule['attn_qkv'][0]) == num_layers
        assert sum(self.repeated_times_schedule['attn_proj'][0]) == num_layers
        
        self.repeated_times_dim = repeated_times_schedule['dim']
        
        self.qkv = RepeatedModuleList(Layer_num=num_layers, sublist_repeated_times=self.repeated_times_schedule['attn_qkv'][0], sublist_with_flag=self.repeated_times_schedule['attn_qkv'][1], Sublist_instance=nn.Linear, Default_instance=nn.Linear, in_dim=self.repeated_times_dim, out_dim=[i * 3 for i in self.repeated_times_dim], bias=qkv_bias)
    
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = RepeatedModuleList(Layer_num=num_layers, sublist_repeated_times=self.repeated_times_schedule['attn_proj'][0], sublist_with_flag=self.repeated_times_schedule['attn_proj'][1], Sublist_instance=nn.Linear, Default_instance=nn.Linear, in_dim=self.repeated_times_dim, out_dim=self.repeated_times_dim)
        
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N, 3, C // self.head_dim, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        out = attn @ v

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SWS_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_paths=[0.], act_layer=nn.GELU, repeated_times_schedule=None, num_layers = 12):
        super().__init__()
        assert isinstance(repeated_times_schedule, dict)
        self.repeated_times_schedule = repeated_times_schedule

        self.repeated_times_dim = repeated_times_schedule['dim']


        assert sum(self.repeated_times_schedule['norm1'][0]) == num_layers
        self.norm1 = RepeatedModuleList_norm(Layer_num=num_layers, sublist_repeated_times=self.repeated_times_schedule['norm1'][0], sublist_with_flag=self.repeated_times_schedule['norm1'][1],  Sublist_instance=nn.LayerNorm, Default_instance=nn.LayerNorm, normalized_dim = self.repeated_times_dim)

        assert sum(self.repeated_times_schedule['norm2'][0]) == num_layers
        self.norm2 = RepeatedModuleList_norm(Layer_num=num_layers, sublist_repeated_times=self.repeated_times_schedule['norm2'][0], sublist_with_flag=self.repeated_times_schedule['norm2'][1],Sublist_instance=nn.LayerNorm, Default_instance=nn.LayerNorm, normalized_dim = self.repeated_times_dim)

        
        self.attn = SWS_Attention(
            dim=self.repeated_times_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, repeated_times_schedule=repeated_times_schedule, num_layers = num_layers
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_paths = nn.ModuleList([DropPath(drop_path) if drop_path > 0. else nn.Identity() for drop_path in drop_paths])
        
        
        mlp_hidden_dim = [i * mlp_ratio for i in self.repeated_times_dim] 

        self.mlp = SWS_Mlp(in_features=self.repeated_times_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, repeated_times_schedule=repeated_times_schedule, num_layers = num_layers
        )

    def forward(self, x):
        drop_path = self.drop_paths[self._layer_id]
        x = x + drop_path(self.attn(self.norm1(x)))
        x = x + drop_path(self.mlp(self.norm2(x)))
        return x


class RepeatedLEBlock(nn.Module):
    def __init__(self, num_layers: int, repeated_times_schedule: dict, **kwargs):
        super().__init__()
        self.num_layers = num_layers

        self.repeated_times_schedule = repeated_times_schedule

        self.block = SWS_Block(repeated_times_schedule=repeated_times_schedule, num_layers = num_layers,**kwargs)

        
        def set_num_layers_fn(m):
            m._num_layers = num_layers
        self.apply(set_num_layers_fn)
            

    def forward(self, x):
        lt = 0
        for i, t in enumerate(range(self.num_layers)):
            def set_layer_id(m):
                m._layer_id = i
                if hasattr(m, 'layer_id_2_repeated_times'):
                    m.layer_id_2_repeated_times()

            self.block.apply(set_layer_id)
            x = self.block(x)

            if self.repeated_times_schedule['transform_flag'][i]:
                x = self.linear_transform[lt](x)
                lt = lt + 1 
        return x

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(num_layers={self.num_layers})'
        return msg


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=[192, 384, 768], depth=12, num_heads=[3, 6, 12], mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_cls_token=True, repeated_times_schedule=None, use_transform=False, **kwargs):
        
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        assert isinstance(repeated_times_schedule, dict)
        
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim[0])
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0])
            
        num_patches = self.patch_embed.num_patches
        
        # note dimension of cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[0]))
        else:
            self.cls_token = None
        
        pos_embed_len = 1 + num_patches if use_cls_token else num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_len, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        block_kwargs = dict(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate
        )
        
        # block
        self.blocks = RepeatedLEBlock(
            drop_paths=dpr,
            num_layers=depth, 
            repeated_times_schedule=repeated_times_schedule,
            **block_kwargs,
        )


        self.norm = nn.LayerNorm(embed_dim[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        if not use_cls_token:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        else:
            self.avgpool = None

        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.apply(self._init_custom_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            m.reset_parameters()


    def _init_custom_weights(self, m):
        if hasattr(m, 'init_weights'):
            m.init_weights()


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def get_classifier(self):
        return self.head


    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x)
        if self.cls_token is not None:
            return x[:, 0]
        else:
            return x


    def forward(self, x):
        x = self.forward_features(x)
        if self.avgpool is not None:
            x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
            x = torch.flatten(x, 1)
        x = self.head(x)
        return x

