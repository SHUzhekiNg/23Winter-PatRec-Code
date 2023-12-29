import warnings
warnings.filterwarnings('ignore')

import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font="Arial", rc={"axes.unicode_minus":False})

import torch
from ultralytics import YOLO

from PIL import Image

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# # model = YOLO("./yolov8l.pt")
# model = YOLO("./yolov8n.pt")

# results = model.train(data="./datasets/data.yaml", epochs=100, save_period=10, seed=seed)

# Image.open("./runs/detect/train/labels.jpg")
# Image.open("./runs/detect/train/labels_correlogram.jpg")
# Image.open("./runs/detect/train/PR_curve.png")
# Image.open("./runs/detect/train/confusion_matrix.png")
# Image.open("./runs/detect/train/results.png")
model_best = YOLO("./runs/detect/train/weights/best.pt")
# metrics = model_best.val()

# print("precision(B): ", metrics.results_dict["metrics/precision(B)"])
# print("metrics/recall(B): ", metrics.results_dict["metrics/recall(B)"])
# print("metrics/mAP50(B): ", metrics.results_dict["metrics/mAP50(B)"])
# print("metrics/mAP50-95(B): ", metrics.results_dict["metrics/mAP50-95(B)"])

# Image.open("./runs/detect/val/PR_curve.png")
# Image.open("./runs/detect/val/confusion_matrix.png")

paths = glob.glob("./datasets/test/images/*")

n = 10
results = model_best.predict(paths[:n])
for i in range(n):
    r = results[i]
    img = Image.fromarray(r.plot())
    img.save("img_{}.jpg".format(i))
    # plt.figure(dpi=100)
    # plt.imshow(img)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
