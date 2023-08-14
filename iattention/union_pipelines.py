import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionPipeline

from . import (
    IAttentionSDCPipeline,
    IAttentionKandinskiyV22PriorPipeline,
    IAttentionKandinskiyV22ControlnetPipeline,
)
from .controlnet_processors import CONTROLNET_PROCESSORS
from .attention_processors import (
    register_stablediffuion_attention_control,
    register_kandinskiy2_2_attention_control,
)


class IAttentionSDCUnionPieline:
    def __init__(self, config):
        self.config = config
        self.prepare_models()
        self.register_attention_colntrol()

    def prepare_models(self):
        controlnet_info = CONTROLNET_PROCESSORS[self.config['common']['controlnet_processor']]

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_info['controlnet'], torch_dtype=torch.float16)
        
        if controlnet_info['is_custom']:
            self.processor = controlnet_info['processor'](
                **controlnet_info['processor_params'])
        else:
            self.processor = controlnet_info['processor'].from_pretrained(
                'lllyasviel/Annotators')

        pipe = IAttentionSDCPipeline.from_pretrained(
            self.config['common']['model_name'],
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to('cuda')

        if self.config['common']['unet_from'] is not None:
            d_pipe = StableDiffusionPipeline.from_single_file(
                self.config['common']['unet_from'],
                torch_dtype=torch.float16,
            )
            pipe.unet = d_pipe.unet.to('cuda')
            del d_pipe

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe

    def register_attention_colntrol(self):
        self.pipe.set_storage_params(self.config['pipe_config'])
        register_stablediffuion_attention_control(
            self.pipe.unet,
            **self.config['unet_config']
        )
        register_stablediffuion_attention_control(
            self.controlnet,
            **self.config['controlnet_config']
        )
        
    def __call__(self, image):
        condition_image = self.processor(Image.fromarray(image))
        result_img = self.pipe(
            image=condition_image,
            prompt=self.config['common']['prompt'],
            negative_prompt=self.config['common']['negative_prompt'],
            num_inference_steps=self.config['common']['num_inference_steps'],
            generator=torch.manual_seed(self.config['common']['seed']),
            guess_mode=self.config['common']['guess_mode'],
        ).images[0]
        result = np.array(result_img)
        return result

    
class IAttentionKandinskiyV22ControlnetUnionPieline:
    def __init__(self, config):
        self.config = config
        self.prepare_models()
        self.register_attention_colntrol()

    def prepare_models(self):
        controlnet_info = CONTROLNET_PROCESSORS[self.config['common']['controlnet_processor']]
        if controlnet_info['is_custom']:
            self.processor = controlnet_info['processor'](
                **controlnet_info['processor_params'])
        else:
            self.processor = controlnet_info['processor'].from_pretrained(
                'lllyasviel/Annotators')
            
        self.pipe_prior = IAttentionKandinskiyV22PriorPipeline.from_pretrained(
            self.config['common']['prior_name'], torch_dtype=torch.float16
        ).to('cuda')

        self.pipe = IAttentionKandinskiyV22ControlnetPipeline.from_pretrained(
            self.config['common']['model_name'], torch_dtype=torch.float16
        ).to('cuda')
    
    def register_attention_colntrol(self):
        self.pipe.set_storage_params(self.config['pipe_config'])
        self.pipe_prior.set_storage_params(self.config['prior_config'])

        register_kandinskiy2_2_attention_control(
            self.pipe.unet,
            **self.config['unet_config']
        )

    def __call__(self, input_image):
        image = Image.fromarray(input_image)
        condition_image = np.array(self.processor(image))
        detected_map = torch.from_numpy(condition_image).float() / 255.0
        hint = detected_map.permute(2, 0, 1).unsqueeze(0).half().to('cuda')

        generator = torch.Generator(device='cuda').manual_seed(self.config['common']['seed'])
        img_emb = self.pipe_prior(
            prompt=self.config['common']['prompt'], 
            image=image, 
            strength=self.config['common']['prior_prompt_strength'], 
            generator=generator
        )
        negative_emb = self.pipe_prior(
            prompt=self.config['common']['negative_prompt'], 
            image=image, 
            strength=self.config['common']['negative_prompt_prior_strength'],
            generator=generator
        )

        result = self.pipe(
            image=image,
            strength=self.config['common']['controlnet_strength'],
            image_embeds=img_emb.image_embeds,
            negative_image_embeds=negative_emb.image_embeds,
            hint=hint,
            num_inference_steps=self.config['common']['num_inference_steps'],
            generator=generator,
            height=self.config['common']['img_h'],
            width=self.config['common']['img_w'],
        ).images[0]

        return np.array(result)

    