# -*- coding: utf-8 -*-
from typing import List, Optional, Union

import torch
import PIL
from diffusers import KandinskyV22PriorEmb2EmbPipeline
from diffusers.pipelines.kandinsky import KandinskyPriorPipelineOutput

from .interpolation_schedulers import INTERPOLATION_SCHEDULERS
from .utils import slerp


class IAttentionKandinskiyV22PriorPipeline(KandinskyV22PriorEmb2EmbPipeline):
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
        prompt: Union[str, List[str]],
        image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]],
        strength: float = 0.3,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 4.0,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif not isinstance(negative_prompt, list) and negative_prompt is not None:
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        # if the negative prompt is defined we double the batch size to
        # directly retrieve the negative prompt embedding
        if negative_prompt is not None:
            prompt = prompt + negative_prompt
            negative_prompt = 2 * negative_prompt

        device = self._execution_device

        batch_size = len(prompt)
        batch_size = batch_size * num_images_per_prompt

        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, text_encoder_hidden_states, text_mask = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        if not isinstance(image, List):
            image = [image]

        if isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

        if isinstance(image, torch.Tensor) and image.ndim == 2:
            # allow user to pass image_embeds directly
            image_embeds = image.repeat_interleave(num_images_per_prompt, dim=0)
        elif isinstance(image, torch.Tensor) and image.ndim != 4:
            raise ValueError(
                f" if pass `image` as pytorch tensor, or a list of pytorch tensor, please make sure each tensor has shape [batch_size, channels, height, width], currently {image[0].unsqueeze(0).shape}"
            )
        else:
            image_embeds = self._encode_image(image, device, num_images_per_prompt)

        # prior
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        latents = image_embeds
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size)
        latents = self.prepare_latents(
            latents,
            latent_timestep,
            batch_size // num_images_per_prompt,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=prompt_embeds,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding

            if do_classifier_free_guidance:
                predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                predicted_image_embedding = predicted_image_embedding_uncond + guidance_scale * (
                    predicted_image_embedding_text - predicted_image_embedding_uncond
                )

            if i + 1 == timesteps.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = timesteps[i + 1]

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample
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
                
        latents = self.prior.post_process_latents(latents)

        image_embeddings = latents

        # if negative prompt has been defined, we retrieve split the image embedding into two
        if negative_prompt is None:
            zero_embeds = self.get_zero_embed(latents.shape[0], device=latents.device)
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()
        else:
            image_embeddings, zero_embeds = image_embeddings.chunk(2)
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.prior_hook.offload()

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            image_embeddings = image_embeddings.cpu().numpy()
            zero_embeds = zero_embeds.cpu().numpy()

        if not return_dict:
            return (image_embeddings, zero_embeds)

        return KandinskyPriorPipelineOutput(image_embeds=image_embeddings, negative_image_embeds=zero_embeds)
