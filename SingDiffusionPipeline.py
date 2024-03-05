from diffusers import UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import os


class SingDiffusionPipeline():
    def __init__(self, sing_diffusion_path, sd15_path, stable_diffusion_pipe, device="cuda"):
        self.device = device
        self.stable_diffusion_pipe = stable_diffusion_pipe
        self.dtype = self.stable_diffusion_pipe.unet.dtype

        # Create sing_diffusion based on UNet2DConditionModel.
        self.sing_diffusion = UNet2DConditionModel.from_config(os.path.join(sing_diffusion_path, 'config.json'))

        # Get rid of the time condition
        self.get_rid_of_time(self.sing_diffusion)

        # Load pretrained SingDiffusion module
        state_dict = torch.load(os.path.join(sing_diffusion_path, 'diffusion_pytorch_model.bin'), map_location="cpu")
        self.sing_diffusion, _, _, _, _ = self.sing_diffusion._load_pretrained_model(self.sing_diffusion, state_dict, None, None)
        self.sing_diffusion = self.sing_diffusion.eval().to(self.device).to(self.dtype)

        # Tokenizer and encoder of SingDiffusion are loaded from SD1.5
        self.text_tokenizer = CLIPTokenizer.from_pretrained(
            sd15_path,
            subfolder="tokenizer", revision=None
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd15_path,
            subfolder="text_encoder", revision=None
        ).to(self.device)

    def get_rid_of_time(self, model):
        for module in self.torch_dfs(model):
            if isinstance(module, ResnetBlock2D):
                module.time_emb_proj = None
                module.time_embedding_norm = None

    def torch_dfs(self, model):
        result = [model]
        for child in model.children():
            result += self.torch_dfs(child)
        return result

    def text_embedding(self, prompt, prompt_uncond=""):
        text_inputs = self.text_tokenizer(
            prompt,
            padding="max_length",
            max_length=self.text_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=None,
        )

        text_inputs_uncond = self.text_tokenizer(
            prompt_uncond,
            padding="max_length",
            max_length=self.text_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_uncond = text_inputs_uncond.input_ids
        text_embeddings_uncond = self.text_encoder(
            text_input_ids_uncond.to(self.device),
            attention_mask=None,
        )
        return text_embeddings[0].to(self.dtype), text_embeddings_uncond[0].to(self.dtype)

    def __call__(self, prompt, prompt_uncond, num_inference_steps=50, height = None, width = None, guidance_scale = 7.5, num_images_per_prompt = 1):
        height = height or self.stable_diffusion_pipe.unet.config.sample_size
        width = width or self.stable_diffusion_pipe.unet.config.sample_size
        channel = self.stable_diffusion_pipe.unet.config.in_channels

        # Set scheduler for stable diffusion pipeline
        self.stable_diffusion_pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Get time 1-epsilon from stable diffusion pipeline
        self.one_minus_epsilon = (
                    torch.ones((1,)).to(self.device) * self.stable_diffusion_pipe.scheduler.timesteps[0]).to(
            device=self.device, dtype=torch.long)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            if prompt_uncond is None:
                prompt_uncond = ""
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            if prompt_uncond is None:
                prompt_uncond = [""] * batch_size

        # Get prompt embedding for SingDiffusion module
        text_embeddings_cond, text_embeddings_uncond = self.text_embedding(prompt, prompt_uncond)
        bs_embed, seq_len, _ = text_embeddings_cond.shape
        text_embeddings_cond = text_embeddings_cond.repeat(1, num_images_per_prompt, 1)
        text_embeddings_cond = text_embeddings_cond.view(bs_embed * num_images_per_prompt, seq_len, -1)
        text_embeddings_uncond = text_embeddings_uncond.repeat(1, num_images_per_prompt, 1)
        text_embeddings_uncond = text_embeddings_uncond.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Initialize x_1 from Gaussian distribution
        noisy_latents = torch.randn((batch_size * num_images_per_prompt, channel, height, width), dtype=self.dtype).to(self.device)

        # Time is not used in SingDiffusion module. Just initialize a place holder
        place_holder_time = torch.ones((batch_size * num_images_per_prompt,)).to(device=self.device, dtype=torch.long)

        # Predict y_bar for positive prompt
        model_pred = self.sing_diffusion(noisy_latents, place_holder_time, text_embeddings_cond).sample

        if guidance_scale > 1:
            # Predict y_bar for negative prompt
            model_pred_uncond = self.sing_diffusion(noisy_latents, place_holder_time, text_embeddings_uncond).sample
            # Classifier-free guidance and normalization
            model_pred = (model_pred_uncond + (model_pred - model_pred_uncond) * guidance_scale) / guidance_scale

        # DDIM sampling process to obtain x_{1-epsilon}
        noisy_latents = self.stable_diffusion_pipe.scheduler.add_noise(model_pred, noisy_latents, self.one_minus_epsilon)

        return noisy_latents