# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionPipeline

from iattention import InterpolationAttentionSDCPipeline
from iattention.attention_processor import create_pipe_attention_control
from iattention.controlnet_processors import CONTROLNET_PROCESSORS
from iattention.utils import correct_colors_hist


def parse_args():
    parser = ArgumentParser(
        description='Process video to video using controlnet and mixing attention.')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/default.yaml',
        required=False,
        help='Path to config with mixing parameters.'
    )
    parser.add_argument(
        '--input_video_path',
        type=str,
        required=False,
        help='Path to original video.',
    )
    parser.add_argument(
        '--output_video_path',
        type=str,
        required=False,
        help='Path to save processed video.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=False,
        help='Condition text for creating images.',
    )
    args = parser.parse_args()
    return args


def process_config(config, args):
    for name, value in dict(vars(args)).items():
        if value is not None:
            config['common'][name] = value

    if 'negative_prompt' not in config['common'] or config['common']['negative_prompt'] is None:
        config['common']['negative_prompt'] = ''

    if 'num_inference_steps' not in config['common'] or config['common']['num_inference_steps'] is None:
        config['common']['num_inference_steps'] = 25

    if 'seed' not in config['common']:
        config['common']['seed'] = 17

    total_steps = config['common']['num_inference_steps']
    config['pipe_config']['total_steps'] = total_steps
    config['unet_config']['total_steps'] = total_steps
    config['controlnet_config']['total_steps'] = total_steps
    return config


def get_video_info(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return height, width, fps, frame_count


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config = process_config(config, args)

    controlnet_info = CONTROLNET_PROCESSORS[config['common']
                                            ['controlnet_processor']]
    controlnet = ControlNetModel.from_pretrained(
        controlnet_info['controlnet'], torch_dtype=torch.float16)
    processor = None
    if controlnet_info['is_custom']:
        processor = controlnet_info['processor'](
            **controlnet_info['processor_params'])
    else:
        processor = controlnet_info['processor'].from_pretrained(
            'lllyasviel/Annotators')

    pipe = InterpolationAttentionSDCPipeline.from_pretrained(
        config['common']['model_name'],
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    ).to('cuda')

    if config['common']['unet_from'] is not None:
        d_pipe = StableDiffusionPipeline.from_single_file(
            config['common']['unet_from'],
            torch_dtype=torch.float16,
        )
        pipe.unet = d_pipe.unet.to('cuda')
        del d_pipe

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    create_pipe_attention_control(
        pipe,
        config['pipe_config'],
        config['unet_config'],
        config['controlnet_config'],
    )

    cap = cv2.VideoCapture(config['common']['input_video_path'])
    orig_height, orig_width, fps, frame_count = get_video_info(cap)

    print('\n\nRUN WITH PARAMETERS:\n', OmegaConf.to_yaml(config['common']))

    print(
        f'\n[ VIDEO INFO | WxH: {orig_width}x{orig_height} | FPS: {fps} | FRAME COUNT: {frame_count} ]\n')

    img_h, img_w = config['common']['img_h'], config['common']['img_w']
    out_h, out_w = img_h, img_w
    if config['common']['original_output_size']:
        out_h, out_w = orig_height, orig_width

    writer = cv2.VideoWriter(
        config['common']['output_video_path'],
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (out_w, out_h),
    )

    first_image = None
    for _ in tqdm(range(frame_count)):
        c_ret, c_image = cap.read()
        if not c_ret or c_image is None:
            break
        image = cv2.resize(c_image, (img_h, img_w))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        condition_image = processor(Image.fromarray(image))

        result_img = pipe(
            image=condition_image,
            prompt=config['common']['prompt'],
            negative_prompt=config['common']['negative_prompt'],
            num_inference_steps=config['common']['num_inference_steps'],
            generator=torch.manual_seed(config['common']['seed']),
            guess_mode=True,
        ).images[0]

        result = np.array(result_img)

        if first_image is None:
            first_image = result.copy()
        else:
            result = correct_colors_hist(
                first_image, result, config['common']['hist_normalize'])

        result = cv2.resize(result, (out_w, out_h))
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        writer.write(result)

    cap.release()
    writer.release()


if __name__ == '__main__':
    main()
