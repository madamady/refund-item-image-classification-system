import os
import sys

img_dir = "model_data/archive (1)/images"
images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
if not images:
    print("no images found", file=sys.stderr)
    sys.exit(1)

sample = os.path.join(img_dir, images[0])
print(sample)
