# -*- coding: utf-8 -*-
from typing import Callable, List, Optional, Union

import PIL
import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers import KandinskyV22ControlnetImg2ImgPipeline

from .interpolation_schedulers import INTERPOLATION_SCHEDULERS
from .utils import slerp


def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.prepare_image
def prepare_image(pil_image, w=512, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    return image


class IAttentionKandinskiyV22ControlnetPipeline(KandinskyV22ControlnetImg2ImgPipeline):
    def set_storage_params(self, pipe_config):
        self.ema = pipe_config['ema']
        self.start_ema = pipe_config['ema']
        self.eta = pipe_config['eta']
        self.storage = [None for _ in range(pipe_config['total_steps'])]
        self.total_steps = pipe_config['total_steps']
        self.start_step = pipe_config['start_step']
        self.end_step = pipe_config['end_step']
        self.const_steps = pipe_config['const_steps']
        self.interpolation_scheduler = INTERPOLATION_SCHEDULERS[
            pipe_config['interpolation_scheduler']]

    def interpolate_with_storage_latents(self, embedding, cur_step):
        out_embedding = embedding.clone()
        if (self.start_step < cur_step < self.end_step):
            if self.storage[cur_step] is None:
                self.storage[cur_step] = embedding.clone()
            else:
                out_embedding = slerp(
                    self.ema, embedding, self.storage[cur_step])
                self.storage[cur_step] = out_embedding.clone()
        return out_embedding

    @torch.no_grad()
    def __call__(
        self,
        image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]],
        negative_image_embeds: Union[torch.FloatTensor, List[torch.FloatTensor]],
        hint: torch.FloatTensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        strength: float = 0.3,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(image_embeds, list):
            image_embeds = torch.cat(image_embeds, dim=0)
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = torch.cat(negative_image_embeds, dim=0)
        if isinstance(hint, list):
            hint = torch.cat(hint, dim=0)

        batch_size = image_embeds.shape[0]

        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            hint = hint.repeat_interleave(num_images_per_prompt, dim=0)

            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
                dtype=self.unet.dtype, device=device
            )
            hint = torch.cat([hint, hint], dim=0).to(dtype=self.unet.dtype, device=device)

        if not isinstance(image, list):
            image = [image]
        if not all(isinstance(i, (PIL.Image.Image, torch.Tensor)) for i in image):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and pytorch tensor"
            )

        image = torch.cat([prepare_image(i, width, height) for i in image], dim=0)
        image = image.to(dtype=image_embeds.dtype, device=device)

        latents = self.movq.encode(image)["latents"]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        height, width = downscale_height_and_width(height, width, self.movq_scale_factor)
        latents = self.prepare_latents(
            latents, latent_timestep, batch_size, num_images_per_prompt, image_embeds.dtype, device, generator
        )
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            added_cond_kwargs = {"image_embeds": image_embeds, "hint": hint}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]
            latents = self.interpolate_with_storage_latents(latents, i)

            if self.start_step + self.const_steps < i:
                mix_kwargs = {
                    'current_index': i - self.const_steps - self.start_step,
                    'total_steps': self.total_steps - self.const_steps - self.start_step,
                    'start_value': self.start_ema,
                    'eta': self.eta,
                }
                self.ema = self.interpolation_scheduler(**mix_kwargs)

            if i == self.total_steps - 1:
                self.ema = self.start_ema

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # post-processing
        image = self.movq.decode(latents, force_not_quantize=True)["sample"]

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if output_type not in ["pt", "np", "pil"]:
            raise ValueError(f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}")

        if output_type in ["np", "pil"]:
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
