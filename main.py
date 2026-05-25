import os
import sys
import argparse
import datetime
import logging

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.engine import Engine
from model.client import Client
from utils.utils import setSeed, initLogging
from utils.data_loader import MovieLensDataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args():
    p = argparse.ArgumentParser(description='FedSCM')

    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--dataset_name', type=str, default='ml-100k',
                   choices=['ml-100k', 'ml-1m', 'video', 'pet'])
    p.add_argument('--num_users', type=int, default=943)
    p.add_argument('--num_items', type=int, default=1682)

    p.add_argument('--item_emb_init_path', type=str, default=None)

    p.add_argument('--num_round', type=int, default=100)
    p.add_argument('--clients_sample_ratio', type=float, default=1.0)
    p.add_argument('--local_epoch', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_negative', type=int, default=4)

    p.add_argument('--lr_network', type=float, default=0.1)
    p.add_argument('--lr_eta', type=int, default=80)
    p.add_argument('--l2_regularization', type=float, default=0.0)
    p.add_argument('--lambda_delta', type=float, default=0.0)

    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--lora_rank', type=int, default=1)

    p.add_argument('--random_seed', type=int, default=0)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--use_cuda', type=bool, default=True)

    # Optional: add Laplace(0, lambda) noise to client item embeddings before upload
    p.add_argument('--ldp_enable', action='store_true',
                   help='Add Laplace noise to uploaded item embeddings')
    p.add_argument('--ldp_laplace_lambda', type=float, default=0.0,
                   help='Noise strength lambda for Laplace(0, lambda). 0 means no noise.')

    return p.parse_args()


def build_positive_iids(sample_generator) -> dict:
    df = sample_generator.train_data
    out = {}
    for uid, g in df.groupby('uid'):
        pos = g.loc[g['rating'] > 0, 'iid']
        out[int(uid)] = torch.tensor(pos.values, dtype=torch.long)
    return out


if __name__ == '__main__':
    args = get_args()
    setSeed(args.random_seed)

    overrides = {
        'ml-100k': (943, 1682),
        'ml-1m': (6040, 3706),
        'video': (6418, 4338),
        'pet': (8065, 27128),
    }
    args.num_users, args.num_items = overrides.get(args.dataset_name, (args.num_users, args.num_items))

    if args.item_emb_init_path is None:
        args.item_emb_init_path = os.path.join(args.data_dir, args.dataset_name, 'item_32d.npy')

    os.makedirs('logs', exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = f'logs/FedSCM.{args.dataset_name}.D{args.latent_dim}.R{args.lora_rank}.{ts}.txt'
    initLogging(log_file)
    logging.info(args)

    data_path = os.path.join(args.data_dir, args.dataset_name)
    sample_generator = MovieLensDataLoader(
        data_path, args.num_negative, args.batch_size,
        args.num_users, args.num_items, args.dataset_name,
    )
    vali_data = sample_generator.get_vali_data
    test_data = sample_generator.get_test_data

    logging.info('Total params: %d', sum(p.numel() for p in Client(args).parameters()))
    engine = Engine(args)
    engine.set_train_positive_iids(build_positive_iids(sample_generator))

    test_hrs, test_ndcgs = [], []
    val_hrs, val_ndcgs = [], []

    for rnd in range(args.num_round):
        train_data = sample_generator.negative_sampling(args.num_negative)
        losses = engine.fed_train_a_round(train_data)
        avg_loss = sum(losses.values()) / len(losses)

        t_hr, t_ndcg = engine.fed_evaluate(test_data)
        v_hr, v_ndcg = engine.fed_evaluate(vali_data)
        t_hr, t_ndcg = float(t_hr[0]), float(t_ndcg[0])
        v_hr, v_ndcg = float(v_hr[0]), float(v_ndcg[0])

        test_hrs.append(t_hr)
        test_ndcgs.append(t_ndcg)
        val_hrs.append(v_hr)
        val_ndcgs.append(v_ndcg)

        logging.info(
            '[%3d] loss=%.4f | test HR@%d=%.6f NDCG@%d=%.6f | val HR@%d=%.6f NDCG@%d=%.6f',
            rnd + 1, avg_loss,
            args.top_k, t_hr, args.top_k, t_ndcg,
            args.top_k, v_hr, args.top_k, v_ndcg,
        )

    best_hr = int(np.argmax(val_hrs))
    best_ndcg = int(np.argmax(val_ndcgs))
    logging.info('--- Best (by val HR)   R%d  test HR=%.6f  NDCG=%.6f',
                 best_hr + 1, test_hrs[best_hr], test_ndcgs[best_hr])
    logging.info('--- Best (by val NDCG) R%d  test HR=%.6f  NDCG=%.6f',
                 best_ndcg + 1, test_hrs[best_ndcg], test_ndcgs[best_ndcg])
