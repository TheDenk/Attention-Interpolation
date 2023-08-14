# -*- coding: utf-8 -*-
from .stablediffusion_controlnet_pipeline import IAttentionSDCPipeline
from .kandinskiy2_2_prior_pipeline import IAttentionKandinskiyV22PriorPipeline
from .kandinskiy2_2_controlnet_pipeline import IAttentionKandinskiyV22ControlnetPipeline
from .union_pipelines import (
    IAttentionSDCUnionPieline,
    IAttentionKandinskiyV22ControlnetUnionPieline,
)


__version__ = '0.1.0'

__all__ = [
    'IAttentionSDCPipeline', 
    'IAttentionKandinskiyV22PriorPipeline',
    'IAttentionKandinskiyV22ControlnetPipeline',
    'IAttentionSDCUnionPieline',
    'IAttentionKandinskiyV22ControlnetUnionPieline',
]

UNION_PIPELINES = {
    'StableDiffusion': IAttentionSDCUnionPieline,
    'Kandinskiy2_2': IAttentionKandinskiyV22ControlnetUnionPieline,
}