

import os
import numpy as np
from PIL import Image

root = "data/mini"
rng = np.random.default_rng(0)
for split in ("train", "val"):
    for cls in ("cat", "dog"):
        os.makedirs(os.path.join(root, split, cls), exist_ok=True)
        for i in range(8):
            arr = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(root, split, cls, f"{i}.JPEG"))
