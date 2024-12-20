# ComfyUI-GlifNodes

Custom nodes for comfyUI.

# Install

```shell
pip install -r requirements.txt
```

# Nodes

<details>
  <summary><b><code>GlifConsistencyDecoder</code></b> openai's consistency decoder from hf hub</summary>
    <img src="docs/consistency_vae.png" max-height="500px"/>
</details>
<details>
  <summary><b><code>🧪GlifPatchConsistencyDecoderTiled</code></b> decoder supporting tiled decoding</summary>
    <img src="docs/patch_vae.png" max-height="500px"/>
</details>
<details>
  <summary><b><code>SDXLAspectRatio</code></b> find the closest SDXL height and width for an image</summary>
    <img src="docs/sdxl_aspect_ratio.png" max-height="500px"/>
</details>
<details>
  <summary><b><code>ImageToMultipleOf</code></b> either crop or stretch an image to a multiple of a specific value</summary>
    <img src="docs/multiple_of.png" max-height="500px"/>
</details>
<details>
  <summary><b><code>HFHubLoraLoader</code></b> load LoRAs directly from Huggingface</summary>
    <img src="docs/load_hf_lora.png" max-height="500px"/>
</details>
<details>
  <summary><b><code>HFHubEmbeddingLoader</code></b> load embeddings directly from Huggingface</summary>
    <img src="docs/load_hf_embedding.png" max-height="500px"/>
</details>
<details>
  <summary><b><code>LoraLoaderFromURL</code></b> load LoRAs using a url.</summary>
    <img src="docs/load_lora_from_url.png" max-height="500px"/>
    
This will work with Huggingface, Civitai, and possibly others. Most models on Civitai will require an API key
to download, which can be obtained on your [Civitai account page](https://civitai.com/user/account). Add it to your
environment variables as `CIVITAI_API_KEY`.
</details>
<details>
  <summary><b><code>GlifVariable</code></b> easily use glif variables on the glif comfy editor</summary>
    <img src="docs/glif_variable.png" max-height="500px"/>
</details>

<br>

# Acknowledgements

diffusers: https://github.com/huggingface/diffusers
openai - consistencydecoder: https://github.com/openai/consistencydecoder
reddit user Total-Resort-3120 for float ramp node

# dev

Run tests:

```shell
python tests/test_image_padding_advanced.py 
```