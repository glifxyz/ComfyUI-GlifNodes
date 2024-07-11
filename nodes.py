import os
from typing import Tuple

import comfy.sd
import comfy.utils
import torch
import torch.nn.functional as F
from comfy.sd import CLIP
from diffusers import ConsistencyDecoderVAE
import folder_paths
from huggingface_hub import hf_hub_download
from torch import Tensor


def find_or_create_cache():
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
    CATEGORY = "latent"
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    def __init__(self):
        self.vae = (
            ConsistencyDecoderVAE.from_pretrained(
                "openai/consistency-decoder",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                cache_dir=find_or_create_cache(),
            )
            .eval()
            .to("cuda")
        )

    def _decode(self, latent):
        """Used when patching another vae."""
        return self.vae.decode(latent.half().cuda()).sample

    def decode(self, latent):
        """Used for standalone decoding."""
        sample = self._decode(latent["samples"])
        sample = sample.clamp(-1, 1).movedim(1, -1).add(1.0).mul(0.5).cpu()
        return (sample,)


class PatchDecoderTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    category = "vae"

    def __init__(self):
        self.vae = ConsistencyDecoder()

    def patch(self, vae):
        del vae.first_stage_model.decoder
        vae.first_stage_model.decode = self.vae._decode
        vae.decode = (
            lambda x: vae.decode_tiled_(
                x,
                tile_x=512,
                tile_y=512,
                overlap=64,
            )
            .to("cuda")
            .movedim(1, -1)
        )

        return (vae,)


# quick node to set SDXL-friendly aspect ratios in 1024^2
# adapted from throttlekitty
class SDXLAspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, image: Tensor) -> Tuple[int, int]:
        _, height, width, _ = image.shape
        aspect_ratio = width / height

        aspect_ratios = (
            (1 / 1, 1024, 1024),
            (2 / 3, 832, 1216),
            (3 / 4, 896, 1152),
            (5 / 8, 768, 1216),
            (9 / 16, 768, 1344),
            (9 / 19, 704, 1472),
            (9 / 21, 640, 1536),
            (3 / 2, 1216, 832),
            (4 / 3, 1152, 896),
            (8 / 5, 1216, 768),
            (16 / 9, 1344, 768),
            (19 / 9, 1472, 704),
            (21 / 9, 1536, 640),
        )

        # find the closest aspect ratio
        closest = min(aspect_ratios, key=lambda x: abs(x[0] - aspect_ratio))

        return (closest[1], closest[2])


class ImageToMultipleOf:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple_of": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 256,
                        "step": 16,
                        "display": "number",
                    },
                ),
                "method": (["center crop", "rescale"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, image: Tensor, multiple_of: int, method: str) -> Tuple[Tensor]:
        """Center crop the image to a specific multiple of a number."""
        _, height, width, _ = image.shape

        new_height = height - (height % multiple_of)
        new_width = width - (width % multiple_of)

        if method == "rescale":
            return (
                F.interpolate(
                    image.unsqueeze(0),
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0),
            )
        else:
            top = (height - new_height) // 2
            left = (width - new_width) // 2
            bottom = top + new_height
            right = left + new_width
            return (image[:, top:bottom, left:right, :],)


class HFHubLoraLoader:
    CATEGORY = "loaders"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "repo_id": ("STRING", {"default": ""}),
                "subfolder": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": ""}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    def __init__(self):
        self.loaded_lora = None
        self.loaded_lora_path = None

    def load_lora(
        self,
        model,
        clip,
        repo_id: str,
        subfolder: str,
        filename: str,
        strength_model: float,
        strength_clip: float,
    ):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = hf_hub_download(
            repo_id=repo_id.strip(),
            subfolder=None
            if subfolder is None or subfolder.strip() == ""
            else subfolder.strip(),
            filename=filename.strip(),
            cache_dir=find_or_create_cache(),
        )

        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora_path == lora_path:
                lora = self.loaded_lora
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp
                self.loaded_lora_path = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = lora
            self.loaded_lora_path = lora_path

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


class HFHubEmbeddingLoader:
    """Load a text model embedding from Huggingface Hub.
    The connected CLIP model is not manipulated."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "repo_id": ("STRING", {"default": ""}),
                "subfolder": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "download_embedding"

    CATEGORY = "n/a"

    def download_embedding(
        self,
        clip: CLIP,  # added to signify it's best put in between nodes
        repo_id: str,
        subfolder: str,
        filename: str,
    ):
        hf_hub_download(
            repo_id=repo_id.strip(),
            subfolder=None
            if subfolder is None or subfolder.strip() == ""
            else subfolder.strip(),
            filename=filename.strip(),
            local_dir=folder_paths.get_folder_paths("embeddings")[0],
        )

        return (clip,)


class GlifVariable:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "variable": (
                    [
                        "",
                    ],
                ),
                "fallback": (
                    "STRING",
                    {
                        "default": "",
                        "single_line": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    FUNCTION = "do_it"

    CATEGORY = "glif/variables"

    @classmethod
    def VALIDATE_INPUTS(cls, variable: str, fallback: str):
        # Since we populate dynamically, comfy will report invalid inputs. Override to always return True
        return True

    def do_it(self, variable: str, fallback: str):
        variable = variable.strip()
        fallback = fallback.strip()
        if variable == "" or (variable.startswith("{") and variable.endswith("}")):
            variable = fallback

        int_val = 0
        float_val = 0.0
        string_val = f"{variable}"
        try:
            int_val = int(variable)
        except Exception as _:
            pass
        try:
            float_val = float(variable)
        except Exception as _:
            pass
        return (string_val, int_val, float_val)


class FilmGrainNode:
    CATEGORY = "image/postprocessing"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.5, 
                    "max": 5.0, 
                    "step": 0.1
                }),
                "grain_saturation": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "brightness_impact": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "image_saturation": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.01
                }),
                "mode": (["Color", "Black and White"],),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_film_grain"

    def apply_film_grain(self, image, intensity, grain_size, grain_saturation, brightness_impact, image_saturation, mode):
        device = image.device
        batch_size, height, width, channels = image.shape

        # Generate base noise
        noise_h, noise_w = max(1, int(height / grain_size)), max(1, int(width / grain_size))
        
        # Generate monochrome noise
        mono_noise = torch.randn(batch_size, noise_h, noise_w, 1, device=device)

        # Generate color noise
        color_noise = torch.randn(batch_size, noise_h, noise_w, channels, device=device)

        # Blend monochrome and color noise based on saturation
        noise = mono_noise.repeat(1, 1, 1, channels) * (1 - grain_saturation) + color_noise * grain_saturation

        # Resize noise if necessary
        if grain_size != 1.0:
            noise = torch.nn.functional.interpolate(noise.permute(0, 3, 1, 2), 
                                                    size=(height, width), 
                                                    mode='bilinear', 
                                                    align_corners=False)
            noise = noise.permute(0, 2, 3, 1)

        # Calculate brightness for brightness-dependent grain
        brightness = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # Modulate noise intensity based on brightness
        brightness_factor = (1 - brightness_impact) + brightness_impact * brightness.unsqueeze(-1)
        
        # Apply noise with brightness adjustment
        grain = (noise - 0.5) * intensity * brightness_factor
        grainy_img = image + grain

        # Apply image saturation adjustment in color mode
        if mode == "Color":
            # Convert to HSV
            r, g, b = grainy_img[..., 0], grainy_img[..., 1], grainy_img[..., 2]
            max_rgb, _ = torch.max(grainy_img, dim=-1)
            min_rgb, _ = torch.min(grainy_img, dim=-1)
            diff = max_rgb - min_rgb

            # Value
            v = max_rgb

            # Saturation
            s = torch.where(v != 0, diff / v, torch.zeros_like(v))

            # Hue
            h = torch.zeros_like(s)
            h[r == v] = (60 * (g[r == v] - b[r == v]) / diff[r == v] % 360) / 360
            h[g == v] = (120 + 60 * (b[g == v] - r[g == v]) / diff[g == v]) / 360
            h[b == v] = (240 + 60 * (r[b == v] - g[b == v]) / diff[b == v]) / 360

            # Adjust saturation
            s = torch.clamp(s * image_saturation, 0, 1)

            # Convert back to RGB
            c = v * s
            x = c * (1 - torch.abs((h * 6) % 2 - 1))
            m = v - c

            rgb = torch.zeros_like(grainy_img)
            mask = (h < 1/6)
            rgb[mask] = torch.stack([c[mask], x[mask], torch.zeros_like(x[mask])], dim=-1)
            mask = (1/6 <= h) & (h < 2/6)
            rgb[mask] = torch.stack([x[mask], c[mask], torch.zeros_like(x[mask])], dim=-1)
            mask = (2/6 <= h) & (h < 3/6)
            rgb[mask] = torch.stack([torch.zeros_like(x[mask]), c[mask], x[mask]], dim=-1)
            mask = (3/6 <= h) & (h < 4/6)
            rgb[mask] = torch.stack([torch.zeros_like(x[mask]), x[mask], c[mask]], dim=-1)
            mask = (4/6 <= h) & (h < 5/6)
            rgb[mask] = torch.stack([x[mask], torch.zeros_like(x[mask]), c[mask]], dim=-1)
            mask = (5/6 <= h)
            rgb[mask] = torch.stack([c[mask], torch.zeros_like(x[mask]), x[mask]], dim=-1)

            grainy_img = rgb + m.unsqueeze(-1)

        elif mode == "Black and White":
            grainy_img = 0.299 * grainy_img[..., 0] + 0.587 * grainy_img[..., 1] + 0.114 * grainy_img[..., 2]
            grainy_img = grainy_img.unsqueeze(-1).repeat(1, 1, 1, 3)

        grainy_img = torch.clamp(grainy_img, 0, 1)

        return (grainy_img,)

NODE_CLASS_MAPPINGS = {
    "GlifConsistencyDecoder": ConsistencyDecoder,
    "GlifPatchConsistencyDecoderTiled": PatchDecoderTiled,
    "SDXLAspectRatio": SDXLAspectRatio,
    "ImageToMultipleOf": ImageToMultipleOf,
    "HFHubLoraLoader": HFHubLoraLoader,
    "HFHubEmbeddingLoader": HFHubEmbeddingLoader,
    "GlifVariable": GlifVariable,
    "FilmGrain": FilmGrainNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GlifConsistencyDecoder": "Consistency VAE Decoder",
    "GlifPatchConsistencyDecoderTiled": "Patch Consistency VAE Decoder",
    "SDXLAspectRatio": "Image to SDXL compatible WH",
    "ImageToMultipleOf": "Image to Multiple of",
    "HFHubLoraLoader": "Load HF Lora",
    "HFHubEmbeddingLoader": "Load HF Embedding",
    "GlifVariable": "Glif Variable",
    "FilmGrain": "Film Grain Effect"
}
