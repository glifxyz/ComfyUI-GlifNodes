import os
import torch
from diffusers import ConsistencyDecoderVAE


def find_and_maybe_create_cache():
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "ComfyUI")):
        cwd = os.path.join(cwd, "ComfyUI")
    if os.path.exists(os.path.join(cwd, "models")):
        cwd = os.path.join(cwd, "models")
    if not os.path.exists(os.path.join(cwd, "huggingface_cache")):
        print("Creating huggingface_cache directory within comfy")
        os.mkdir(os.path.join(cwd, "huggingface_cache"))
    
    return str(os.path.join(cwd, "huggingface_cache"))


class ConsistencyDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def __init__(self):
        self.vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=find_and_maybe_create_cache(),
        ).to("cuda")
        
    def decode(self, latent):    
        print(latent["samples"].shape)
        sample = self.vae.decode(latent["samples"].half().cuda()).sample
        sample = sample.clamp(-1, 1).movedim(1,-1).add(1.).mul(0.5).cpu()
        return (sample, )


NODE_CLASS_MAPPINGS = {
    "GlifConsistencyVAE": ConsistencyDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GlifConsistencyVAE": "Consistency VAE Decoder",
}
