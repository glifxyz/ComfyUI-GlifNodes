import os
from typing import Tuple

import torch
import torch.nn.functional as F
from diffusers import ConsistencyDecoderVAE
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
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

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
            return (F.interpolate(
                image.unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0),)
        else:
            top = (height - new_height) // 2
            left = (width - new_width) // 2
            bottom = top + new_height
            right = left + new_width
            return (image[:, top:bottom, left:right, :],)


NODE_CLASS_MAPPINGS = {
    "GlifConsistencyDecoder": ConsistencyDecoder,
    "GlifPatchConsistencyDecoderTiled": PatchDecoderTiled,
    "SDXLAspectRatio": SDXLAspectRatio,
    "ImageToMultipleOf": ImageToMultipleOf,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GlifConsistencyDecoder": "Consistency VAE Decoder",
    "GlifPatchConsistencyDecoderTiled": "Patch Consistency VAE Decoder",
    "SDXLAspectRatio": "Image to SDXL compatible WH",
    "ImageToMultipleOf": "Image to Multiple of",
}
