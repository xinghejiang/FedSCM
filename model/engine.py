import copy
import os
import random
import logging

import numpy as np
import torch
import torch.nn as nn

from model.client import Client
from utils.utils import compute_metrics, ldp_perturb_laplace


def _is_private(key: str) -> bool:
    return key.startswith('affine_output.') or key in ('lora_A', 'lora_B', 'lora_bias')


class Engine:

    def __init__(self, args):
        self.args = args
        self.num_items = args.num_items
        self.latent_dim = args.latent_dim
        self.lora_rank = getattr(args, 'lora_rank', 4)
        self.epoch = args.local_epoch

        self.l2_reg = args.l2_regularization
        self.lambda_delta = args.lambda_delta

        lr = args.lr_network
        self.lr_global = lr
        self.lr_item = lr * args.num_items * args.lr_eta
        self.lr_lora = lr

        self.train_positive_iids = {}
        self.client_private_params = {}
        self.current_round = 0

        _base = Client(args)
        self.server_param = {
            'item_emb_global.weight': _base.item_emb_global.weight.data.cpu().clone()
        }
        self._load_pretrained_emb(args)

        self.crit = nn.BCELoss()
        self.device = self._resolve_device(args)

        self.model = Client(args).to(self.device)

    def _load_pretrained_emb(self, args):
        path = getattr(args, 'item_emb_init_path', None)
        if path and os.path.isfile(path):
            emb = np.load(path, allow_pickle=True).astype(np.float32)
            if emb.shape == (self.num_items, self.latent_dim):
                self.server_param['item_emb_global.weight'] = torch.from_numpy(emb)
                logging.info('Pretrained embedding loaded: %s', path)

    @staticmethod
    def _resolve_device(args):
        if args.use_cuda:
            if torch.cuda.is_available():
                return torch.device('cuda')
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                return torch.device('mps')
        return torch.device('cpu')

    def set_train_positive_iids(self, mapping: dict):
        self.train_positive_iids = {int(k): v for k, v in mapping.items()}

    def _user_anchor(self, user_id: int) -> torch.Tensor:
        W = self.server_param['item_emb_global.weight']
        iids = self.train_positive_iids.get(user_id)
        if iids is None or len(iids) == 0:
            return torch.zeros(self.latent_dim)
        return W[torch.unique(iids).long()].float().mean(0)

    def _build_model(self, user_id: int) -> Client:
        state = copy.deepcopy(self.model.state_dict())
        state['item_emb_global.weight'] = self.server_param['item_emb_global.weight'].clone()
        if user_id in self.client_private_params:
            for k, v in self.client_private_params[user_id].items():
                state[k] = v.clone()
        for k in state:
            state[k] = state[k].to(self.device)
        m = copy.deepcopy(self.model)
        m.load_state_dict(state, strict=False)
        return m

    def _save_private(self, user_id: int, model: Client):
        sd = model.state_dict()
        self.client_private_params[user_id] = {
            k: sd[k].detach().cpu().clone() for k in sd if _is_private(k)
        }

    def _lora_l2(self, model: Client) -> torch.Tensor:
        return (model.lora_A ** 2).sum() + (model.lora_B ** 2).sum() + (model.lora_bias ** 2).sum()

    def _train_batch(self, model, X, y, opt):
        X, y = X.to(self.device), y.to(self.device)
        opt.zero_grad()
        loss = self.crit(model(X).view(-1), y) + self.lambda_delta * self._lora_l2(model)
        loss.backward()
        opt.step()
        return loss.item()

    def fed_train_a_round(self, train_data: dict) -> dict:
        n_part = max(1, int(self.args.num_users * self.args.clients_sample_ratio))
        users = random.sample(range(self.args.num_users), n_part)

        losses, new_item_weights = {}, []
        ldp_enable = bool(getattr(self.args, 'ldp_enable', False))
        laplace_lambda = float(getattr(self.args, 'ldp_laplace_lambda', 0.0))

        for u in users:
            if u not in train_data:
                continue
            model = self._build_model(u)
            model.set_user_anchor(self._user_anchor(u).to(self.device))
            model.train()

            opt = torch.optim.SGD(
                [
                    {'params': model.affine_output.parameters(), 'lr': self.lr_global},
                    {'params': [model.lora_A, model.lora_B, model.lora_bias], 'lr': self.lr_lora},
                    {'params': model.item_emb_global.parameters(), 'lr': self.lr_item},
                ],
                weight_decay=self.l2_reg,
            )

            total_loss, total_n = 0.0, 0
            for _ in range(self.epoch):
                for X, y in train_data[u]:
                    total_loss += self._train_batch(model, X, y, opt) * len(X)
                    total_n += len(X)

            losses[u] = total_loss / max(1, total_n)
            self._save_private(u, model)

            local_W = model.item_emb_global.weight.detach().cpu()
            if ldp_enable:
                # Default (paper-style): perturb full uploaded item embedding weights.
                noisy_W = ldp_perturb_laplace(local_W, laplace_lambda=laplace_lambda)
                new_item_weights.append(noisy_W)
            else:
                new_item_weights.append(local_W)

        if len(new_item_weights) > 0:
            n = len(new_item_weights)
            acc = None
            for t, w in enumerate(new_item_weights):
                acc = w.clone() if t == 0 else (acc + w)
            self.server_param['item_emb_global.weight'] = (acc / n)
        else:
            logging.warning('No item weights to aggregate; skip FedAvg.')

        self.current_round += 1
        return losses

    def fed_evaluate(self, eval_data: dict):
        results = []
        for u, batches in eval_data.items():
            model = self._build_model(u)
            model.set_user_anchor(self._user_anchor(u).to(self.device))
            model.eval()
            targets, preds = [], []
            with torch.no_grad():
                for X, y in batches:
                    X, y = X.to(self.device), y.to(self.device)
                    p = model(X).squeeze()
                    y = y.squeeze().float()
                    targets.extend(y.tolist() if y.dim() > 0 else [y.item()])
                    preds.extend(p.tolist() if p.dim() > 0 else [p.item()])
            results.append((targets, preds))
        return compute_metrics(results, self.args.top_k)

    def save_snapshot(self, path: str):
        data = {
            'server_param': self.server_param,
            'client_private_params': self.client_private_params,
            'train_positive_iids': self.train_positive_iids,
            'args': self.args,
        }
        torch.save(data, path)
        logging.info('Snapshot saved to %s', path)
