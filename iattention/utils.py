# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from torch.linalg import norm
from torch import FloatTensor, LongTensor, Tensor, Size, lerp, zeros_like


def show_image(image, figsize=(5, 5), cmap=None, title='', xlabel=None, ylabel=None, axis=False):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.show()


def show_images(images, n_rows=1, titles=None, figsize=(5, 5), cmap=None, xlabel=None, ylabel=None, axis=False):
    n_cols = len(images) // n_rows
    if n_rows == n_cols == 1:
        if isinstance(titles, str) or titles is None:
            title = titles
        if isinstance(titles, list):
            title = titles[0]
        show_image(images[0], title=title, figsize=figsize,
                   cmap=cmap, xlabel=xlabel, ylabel=ylabel, axis=axis)
    else:
        titles = titles if isinstance(titles, list) else [
            '' for _ in range(len(images))]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.tight_layout(pad=0.0)
        axes = axes.flatten()
        for index, ax in enumerate(axes):
            ax.imshow(images[index], cmap=cmap)
            ax.set_title(titles[index])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis(axis)
        plt.show()


def clear_noise(image, kernel=(3, 3)):
    img = image.copy()

    e_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    erode = cv2.morphologyEx(img, cv2.MORPH_ERODE, e_kernel)

    c_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    dilate = cv2.morphologyEx(erode, cv2.MORPH_DILATE, c_kernel)

    return dilate


def smash_noise(image, kernel=(3, 3)):
    img = image.copy()

    c_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    dilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, c_kernel)

    e_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    erode = cv2.morphologyEx(dilate, cv2.MORPH_ERODE, e_kernel)

    return erode


def slerp(t: float, v0: FloatTensor, v1: FloatTensor, DOT_THRESHOLD=0.9995):
    '''
    https://gist.github.com/Birch-san/230ac46f99ec411ed5907b0a3d728efa
    '''
    assert v0.shape == v1.shape, 'shapes of v0 and v1 must match'

    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = norm(v0, dim=-1)
    v1_norm: FloatTensor = norm(v1, dim=-1)

    v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: FloatTensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp: LongTensor = ~gotta_lerp

    t_batch_dim_count: int = max(
        0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(
        t, Tensor) else Size([])
    out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))

    # if no elements are lerpable, our vectors become 0-dimensional, preventing broadcasting
    if gotta_lerp.any():
        lerped: FloatTensor = lerp(v0, v1, t)

        out: FloatTensor = lerped.where(gotta_lerp.unsqueeze(-1), out)

    # if no elements are slerpable, our vectors become 0-dimensional, preventing broadcasting
    if can_slerp.any():

        # Calculate initial angle between v0 and v1
        theta_0: FloatTensor = dot.arccos().unsqueeze(-1)
        sin_theta_0: FloatTensor = theta_0.sin()
        # Angle at timestep t
        theta_t: FloatTensor = theta_0 * t
        sin_theta_t: FloatTensor = theta_t.sin()
        # Finish the slerp algorithm
        s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
        s1: FloatTensor = sin_theta_t / sin_theta_0
        slerped: FloatTensor = s0 * v0 + s1 * v1

        out: FloatTensor = slerped.where(can_slerp.unsqueeze(-1), out)

    return out


def correct_colors_hist(cur_img, prev_img, mode):
    '''
    https://github.com/un1tz3r0/controlnetvideo/blob/main/controlnetvideo.py#L467
    '''
    if mode == 'rgb':
        return match_histograms(prev_img, cur_img, channel_axis=2)
    elif mode == 'hsv':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        cur_img_hsv = cv2.cvtColor(cur_img, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(
            prev_img_hsv, cur_img_hsv, channel_axis=2)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    elif mode == 'lab':
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        cur_img_lab = cv2.cvtColor(cur_img, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(
            prev_img_lab, cur_img_lab, channel_axis=2)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
    else:
        raise ValueError('Invalid color mode')
