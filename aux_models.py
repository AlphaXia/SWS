import torch
from timm.models.registry import register_model
from models import *


def get_repeated_schedule_9():
    return {
        'norm1': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'norm2': [[3, 3, 4, 3, 3], [True, True, True, True, True]],  
        'attn_rpe': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'attn_qkv': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'attn_proj': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'mlp_fc1': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'mlp_fc2': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'dim': [768, 768, 768, 768, 768],
        'transform_flag': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }


def get_repeated_schedule_12():
    return {
        'norm1': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'norm2': [[3, 3, 4, 3, 3], [True, True, True, True, True]],  
        'attn_rpe': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'attn_qkv': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'attn_proj': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'mlp_fc1': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'mlp_fc2': [[3, 3, 4, 3, 3], [True, True, True, True, True]], 
        'dim': [384, 384, 384, 384, 384],
        'transform_flag': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }




@register_model
def aux_deit_small_patch16_224_L16_33433(pretrained=False, **kwargs):
    return deit_small_patch16_224_L16(pretrained=pretrained,
                                use_cls_token=False,
                                repeated_times_schedule=get_repeated_schedule_12(),
                                **kwargs)


@register_model
def aux_deit_base_patch16_224_L16_33433(pretrained=False, **kwargs):
    return deit_base_patch16_224_L16(pretrained=pretrained,
                                use_cls_token=False,
                                repeated_times_schedule=get_repeated_schedule_9(),
                                **kwargs)
