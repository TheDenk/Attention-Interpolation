# -*- coding: utf-8 -*-
import torch
from diffusers.models.attention_processor import AttnProcessor

from .interpolation_schedulers import INTERPOLATION_SCHEDULERS
from .utils import slerp


class IAttnProcessor:
    def __init__(self, use_interpolation, name, allow_names, total_steps,
                 start_step, end_step, attention_res,
                 const_steps=0, eta=0.0, ema=0.0, interpolation_scheduler='ema'):
        super().__init__()
        self.eta = eta
        self.ema = ema
        self.name = name
        self.start_ema = ema
        self.allow_names = allow_names
        self.use_interpolation = use_interpolation
        self.storage = [{
            'key': None,
            'query': None,
            'value': None,
            'attention_probs': None,
            'out_linear': None,
        } for _ in range(total_steps)]

        self.cur_step = 0
        self.total_steps = total_steps
        self.start_step = start_step
        self.end_step = end_step
        self.const_steps = const_steps
        self.attention_res = attention_res * attention_res
        self.interpolation_scheduler = INTERPOLATION_SCHEDULERS[interpolation_scheduler]

    def interpolate_with_storage(self, embedding, name):
        out_embedding = embedding.clone()

        if self.use_interpolation[name] and (self.start_step < self.cur_step < self.end_step):
            if self.storage[self.cur_step][name] is None:
                self.storage[self.cur_step][name] = embedding.clone()
            else:
                out_embedding = slerp(
                    self.ema, embedding, self.storage[self.cur_step][name])
                self.storage[self.cur_step][name] = out_embedding.clone()

        return out_embedding
    
class IStableDiffurionAttnProcessor(IAttnProcessor):
    def __call__(self, attn: AttnProcessor, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size=batch_size)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        condition = self.name in self.allow_names and not is_cross
        if condition:
            query = self.interpolate_with_storage(query, 'query')
            key = self.interpolate_with_storage(key, 'key')
            value = self.interpolate_with_storage(value, 'value')

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if condition and attention_probs.shape[-1] <= self.attention_res:
            attention_probs = self.interpolate_with_storage(
                attention_probs, 'attention_probs')

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        if condition:
            hidden_states = self.interpolate_with_storage(
                hidden_states, 'out_linear')

        hidden_states = attn.to_out[1](hidden_states)

        if self.start_step + self.const_steps < self.cur_step:
            mix_kwargs = {
                'current_index': self.cur_step - self.const_steps - self.start_step,
                'total_steps': self.total_steps - self.const_steps - self.start_step,
                'start_value': self.start_ema,
                'eta': self.eta,
            }
            self.ema = self.interpolation_scheduler(**mix_kwargs)

        self.cur_step += 1

        if self.cur_step == self.total_steps:
            self.cur_step = 0
            self.ema = self.start_ema

        
        return hidden_states


class IKandinskiy2_2AttnProcessor(IAttnProcessor):
    def __call__(self, attn: AttnProcessor, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj
        
        condition = self.name in self.allow_names
        if condition:
            query = self.interpolate_with_storage(query, 'query')
            key = self.interpolate_with_storage(key, 'key')  
            value = self.interpolate_with_storage(value, 'value')
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if condition and attention_probs.shape[-1] <= self.attention_res:
            attention_probs = self.interpolate_with_storage(attention_probs, 'attention_probs')
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states)
        if condition:
            hidden_states = self.interpolate_with_storage(hidden_states, 'out_linear')
            
        hidden_states = attn.to_out[1](hidden_states)
        
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        if self.start_step + self.const_steps < self.cur_step:
            mix_kwargs = {
                'current_index': self.cur_step - self.const_steps - self.start_step,
                'total_steps': self.total_steps - self.const_steps - self.start_step,
                'start_value': self.start_ema,
                'eta': self.eta,
            }
            self.ema = self.interpolation_scheduler(**mix_kwargs)

        self.cur_step += 1

        if self.cur_step == self.total_steps:
            self.cur_step = 0
            self.ema = self.start_ema

        return hidden_states


def register_stablediffuion_attention_control(
        model, use_interpolation, allow_names, total_steps,
        start_step, end_step, attention_res, const_steps=0, 
        eta=1.0, ema=0.0, interpolation_scheduler='ema'
    ):
    attn_procs = {}
    for name in model.attn_processors.keys():
        if name.startswith('mid_block'):
            place_in_unet = 'mid'
        elif name.startswith('up_blocks'):
            place_in_unet = 'up'
        elif name.startswith('down_blocks'):
            place_in_unet = 'down'

        attn_procs[name] = IStableDiffurionAttnProcessor(
            use_interpolation,
            name=place_in_unet,
            allow_names=allow_names,
            total_steps=total_steps,
            eta=eta,
            ema=ema,
            start_step=start_step,
            end_step=end_step,
            const_steps=const_steps,
            attention_res=attention_res,
            interpolation_scheduler=interpolation_scheduler,
        )

    model.set_attn_processor(attn_procs)


def register_kandinskiy2_2_attention_control(
        model, use_interpolation, allow_names, total_steps,
        start_step, end_step, attention_res, const_steps=0, 
        eta=1.0, ema=0.0, interpolation_scheduler='ema'
    ):
    attn_procs = {}
    for name in model.attn_processors.keys():
        if name.startswith('mid_block'):
            place_in_unet = 'mid'
        elif name.startswith('up_blocks'):
            place_in_unet = 'up'
        elif name.startswith('down_blocks'):
            place_in_unet = 'down'

        attn_procs[name] = IKandinskiy2_2AttnProcessor(
            use_interpolation,
            name=place_in_unet,
            allow_names=allow_names,
            total_steps=total_steps,
            eta=eta,
            ema=ema,
            start_step=start_step,
            end_step=end_step,
            const_steps=const_steps,
            attention_res=attention_res,
            interpolation_scheduler=interpolation_scheduler,
        )

    model.set_attn_processor(attn_procs)