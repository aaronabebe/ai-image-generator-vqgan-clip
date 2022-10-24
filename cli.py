import argparse
import sys

import torch
from PIL import Image
from PIL.Image import Resampling
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange, tqdm

from CLIP import clip
from image_generator import load_vqgan_model, MakeCutouts, fetch, parse_prompt, Prompt, resize_image, vector_quantize, \
    clamp_with_grad
from instructions import Instructions, InstructionPrompt


def merge_args(args: argparse.Namespace, instruction_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args), **vars(instruction_args))


def override_args(args: argparse.Namespace, override_args: argparse.Namespace):
    [setattr(args, k, v) for k, v in vars(override_args).items() if v is not None]


def load_models(args, device):
    m = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
    p = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
    return m, p


def main(args: argparse.Namespace, model, perceptor, device, instruction_prompt: InstructionPrompt = None):
    if instruction_prompt:
        override_args(args, instruction_prompt.args_override())

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    n_toks = model.quantize.n_e
    f = 2 ** (model.decoder.num_resolutions - 1)
    toksX, toksY = args.size[0] // f, args.size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.init_image:
        pil_image = Image.open(fetch(args.init_image)).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Resampling.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(fetch(path)).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    if instruction_prompt:
        for i, p in tqdm(instruction_prompt.iter()):
            print(i, p)
            train(args, p.save_path(), opt, z, z_min, z_max, z_orig, make_cutouts, normalize, pMs)
    else:
        for i in trange(args.max_iterations):
            train(args, f'example_peter/progress_{i}.png', opt, z, z_min, z_max, z_orig, make_cutouts, normalize, pMs)


def synth(z: torch.Tensor):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


def ascend_txt(args, pMs, z, z_orig, make_cutouts, normalize):
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if args.init_weight:
        result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

    for prompt in pMs:
        result.append(prompt(iii))

    return result


@torch.no_grad()
def checkin(z, save_path, losses):
    # losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    # print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}', flush=True)
    out = synth(z)
    TF.to_pil_image(out[0].cpu()).save(save_path)


def train(args, save_path, opt, z, z_min, z_max, z_orig, make_cutouts, normalize, pMs):
    opt.zero_grad()
    lossAll = ascend_txt(args, pMs, z, z_orig, make_cutouts, normalize)
    checkin(z, save_path, lossAll)
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--instruction_path", dest="instruction_path", type=str, required=True,
        help="Path to the instruction file. Currently only works with .csv files."
    )
    return parser.parse_args()


if __name__ == '__main__':
    default_args = argparse.Namespace(
        prompts=['eruption of a mountain forest'],
        image_prompts=[],
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[400, 300],
        init_image=None,
        init_weight=0.,
        clip_model='ViT-B/32',
        vqgan_config='vqgan_imagenet_f16_16384.yaml',
        vqgan_checkpoint='vqgan_imagenet_f16_16384.ckpt',
        step_size=0.05,
        cutn=64,
        cut_pow=1.,
        display_freq=1,
        max_iterations=50,
        seed=0,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model, perceptor = load_models(default_args, device)

    if len(sys.argv) < 2:
        main(default_args, model, perceptor, device)
    else:
        cmd_args = merge_args(default_args, parse_args())
        instructions = Instructions.from_csv(cmd_args.instruction_path)
        for prompt in instructions.prompts:
            main(cmd_args, model, perceptor, device, prompt)
