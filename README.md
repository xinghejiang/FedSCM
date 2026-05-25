# FedSCM

This folder provides the training code for FedSCM.

## Layout
```
FedSCM/
├── main.py
├── model/          # client model and federated engine
└── utils/          # metrics, logging, MovieLens-style data loader
```

## Requirements

Python 3.8+. Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU builds of PyTorch, follow [pytorch.org](https://pytorch.org/get-started/locally/) (CUDA wheels use their extra index URL).

## Running

Standard training (no noise):

```bash
cd FedSCM
python main.py --data_dir data --dataset_name ml-100k --num_round 100 --random_seed 0
```

All arguments: `python main.py -h`. Logs are written under `logs/`.


https://github.com/xinghejiang/FedSCM