from diffusers import StableDiffusionPipeline
from SingDiffusionPipeline import SingDiffusionPipeline
import torch
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_sing_diffusion', action='store_true', default=False, help="Not use SingDiffusion")
parser.add_argument('--out_dir', default='./average_brightness_result', help="Output dictionary")
parser.add_argument('--sd15_path', default='runwayml/stable-diffusion-v1-5', help="Stable diffusion 1.5 for text tokenizer and encoder for SingDiffusion")
parser.add_argument('--sd_model_path', default='runwayml/stable-diffusion-v1-5', help="Pre-trained base diffusion model")
parser.add_argument('--sing_diffusion_path', default='./SingDiffusion', help="Pre-trained SingDiffusion")
args = parser.parse_args()

prompt_uncond = None
device = "cuda"
num_inference_steps = 50
num_images_per_prompt = 1

stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_path, torch_dtype=torch.float16).to(device)
sing_diffusion_pipe = SingDiffusionPipeline(args.sing_diffusion_path, args.sd15_path, stable_diffusion_pipe, device=device)

with open('./COCO_3W_prompt.json') as fr:
    prompts_dict = json.load(fr)

image_id = prompts_dict.keys()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

with torch.no_grad():
    for pi, prompt_dic in enumerate(image_id):
        if os.path.exists(os.path.join(args.out_dir, f"{pi:03d}_{prompt_dic}.jpg")):
            pass
        else:
            prompt_uncond = None if prompt_uncond == None else prompt_uncond[i]
            noisy_latents = None
            if not args.no_sing_diffusion:
                # initial singularity time-step sampling
                noisy_latents = sing_diffusion_pipe(prompts_dict[prompt_dic], prompt_uncond, num_inference_steps=num_inference_steps,
                                                num_images_per_prompt=num_images_per_prompt)
            # original diffusion model sampling
            image = stable_diffusion_pipe(prompts_dict[prompt_dic], negative_prompt=prompt_uncond, latents=noisy_latents,
                                          num_inference_steps=num_inference_steps,
                                          num_images_per_prompt=num_images_per_prompt).images[0]
            image.save(os.path.join(args.out_dir, f"{pi:03d}_{prompt_dic}.jpg"))
