# 23Winter-PatRec-Code
## Preparation
Dataset from: https://www.kaggle.com/datasets/pkdarabi/road-mark-detection/data
Checkpoint from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
And place the dataset as below:

```
$ tree . -d

.
├── datasets
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── valid
│       ├── images
│       └── labels
└── yolov8n.pt
```
## Dependencies
1. torch>=1.8.0 from https://pytorch.org/get-started/previous-versions/
2. `pip install ultralytic`


## Run
`python3 run.py`