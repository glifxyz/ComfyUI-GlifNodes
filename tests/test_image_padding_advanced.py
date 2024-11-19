import sys

sys.path.append("./")
sys.path.append("../../")

import numpy as np
import torch
from PIL import Image

from nodes import ImagePaddingAdvanced


def test_image_padding_advanced():
    test_image = Image.open("./docs/glif_variable.png").convert("RGB")
    test_image = torch.from_numpy(np.array(test_image)).unsqueeze(0).permute(0, 1, 2, 3)

    node = ImagePaddingAdvanced()

    target_width = 800
    target_height = 800

    result = node.run(test_image, target_width, target_height, "constant")[0]
    result = result.squeeze(0).cpu().numpy()
    result = Image.fromarray(result.astype(np.uint8))
    result.save("./tests/glif_variable_constant.png")

    result = node.run(test_image, target_width, target_height, "replicate")[0]
    result = result.squeeze(0).cpu().numpy()
    result = Image.fromarray(result.astype(np.uint8))
    result.save("./tests/glif_variable_replicate.png")

    result = node.run(test_image, target_width, target_height, "reflect")[0]
    result = result.squeeze(0).cpu().numpy()
    result = Image.fromarray(result.astype(np.uint8))
    result.save("./tests/glif_variable_reflect.png")

if __name__ == "__main__":
    test_image_padding_advanced()
