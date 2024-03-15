# ComfyUI-GlifNodes

Custom nodes for comfyUI.

# Install

```
pip install -r requirements.txt
```

# Nodes

<details>
  <summary><b><code>GlifConsistencyDecoder</code></b> openai's consistency decoder from hf hub</summary>
    <img src="docs/consistency_vae.png" height="500px"/>
</details>
<details>
  <summary><b><code>🧪GlifPatchConsistencyDecoderTiled</code></b> decoder supporting tiled decoding</summary>
    <img src="docs/patch_vae.png" height="500px"/>
</details>
<details>
  <summary><b><code>SDXLAspectRatio</code></b> find the closest SDXL height and width for an image</summary>
    <img src="docs/sdxl_aspect_ratio.png" height="500px"/>
</details>
<details>
  <summary><b><code>ImageToMultipleOf</code></b> either crop or stretch an image to a multiple of a specific value</summary>
    <img src="docs/multiple_of.png" height="500px"/>
</details>

<br>

# Acknowledgements

diffusers: https://github.com/huggingface/diffusers
openai - consistencydecoder: https://github.com/openai/consistencydecoder
