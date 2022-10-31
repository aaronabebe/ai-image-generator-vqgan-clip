import argparse
import io
import sys

from utils import synth

sys.path.append('./taming-transformers')

import torch
from omegaconf import OmegaConf
import requests
from taming.models import cond_transformer, vqgan

from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from CLIP import clip


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 3)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


def merge_args(args: argparse.Namespace, instruction_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args), **vars(instruction_args))


def override_args(args: argparse.Namespace, override_args: argparse.Namespace):
    [setattr(args, k, v) for k, v in vars(override_args).items() if v is not None]


def load_models(args, device):
    m = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    p = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    return m, p


def ascend_txt(args, model, perceptor, pMs, z, z_orig, make_cutouts, normalize):
    out = synth(model, z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if args.init_weight:
        result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

    for prompt in pMs:
        result.append(prompt(iii))

    return result


@torch.no_grad()
def checkin(model, z, save_path):
    out = synth(model, z)
    TF.to_pil_image(out[0].cpu()).save(save_path)


def train(args, model, perceptor, save_path, opt, z, z_min, z_max, z_orig, make_cutouts, normalize, pMs):
    opt.zero_grad()
    lossAll = ascend_txt(args, model, perceptor, pMs, z, z_orig, make_cutouts, normalize)
    checkin(model, z, save_path, lossAll)
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))
