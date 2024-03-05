from diffusers import StableDiffusionPipeline
from SingDiffusionPipeline import SingDiffusionPipeline
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_sing_diffusion', action='store_true', default=False, help="Not use SingDiffusion")
parser.add_argument('--out_dir', default='./average_brightness_result', help="Output dictionary")
parser.add_argument('--sd15_path', default='runwayml/stable-diffusion-v1-5', help="Stable diffusion 1.5 for text tokenizer and encoder for SingDiffusion")
parser.add_argument('--sd_model_path', default='runwayml/stable-diffusion-v1-5', help="Pre-trained base diffusion model")
parser.add_argument('--sing_diffusion_path', default='./SingDiffusion', help="Pre-trained SingDiffusion")
args = parser.parse_args()

prompt = ["solid black background", "solid white background", "Monochrome line-art logo on a black background", "Monochrome line-art logo on a white background"]
prompt_uncond = None
device = "cuda"
num_inference_steps = 50
num_images_per_prompt = 100

stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_path, torch_dtype=torch.float16).to(device)
sing_diffusion_pipe = SingDiffusionPipeline(args.sing_diffusion_path, args.sd15_path, stable_diffusion_pipe, device=device)

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

for i in range(len(prompt)):
    for j in range(num_images_per_prompt):
        prompt_uncond = None if prompt_uncond == None else prompt_uncond[i]
        noisy_latents = None
        if not args.no_sing_diffusion:
        	# initial singularity time-step sampling
            noisy_latents = sing_diffusion_pipe(prompt[i], prompt_uncond, num_inference_steps=num_inference_steps)
        # original diffusion model sampling
        image = stable_diffusion_pipe(prompt[i], negative_prompt=prompt_uncond, latents=noisy_latents,
                                      num_inference_steps=num_inference_steps).images
        image[0].save(os.path.join(args.out_dir, prompt[i]+'_'+str(j)+'.png'))