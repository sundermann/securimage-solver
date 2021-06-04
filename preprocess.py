import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

def preprocess_image(filename, out):
    im = Image.open(filename)
    im = im.convert("L")

    data = np.array(im)
    data[data > 145] = 255
    data[data <= 145] = 0
    im = Image.fromarray(data)

    im = ImageOps.invert(im)
    imageBox = im.getbbox()
    im = im.crop(imageBox)
    im = ImageOps.invert(im)

    im.save(out)


labels = pd.read_csv("data/labels_new.csv", sep=';', header=None)
def get_label(filename):
    return labels.iat[filename-1, 1]


if __name__ == '__main__':
    for i in range(1, 100000):
        if i < 80000:
            out = "preprocessing/train/{}".format(get_label(i))
            Path(out).mkdir(parents=True, exist_ok=True)
        else:
            out = "preprocessing/test/{}".format(get_label(i))
            Path(out).mkdir(parents=True, exist_ok=True)
        preprocess_image("data/{}.png".format(i), "{}/{}.png".format(out, i))
