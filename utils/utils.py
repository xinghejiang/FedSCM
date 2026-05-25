import torch
import random
import logging
import numpy as np


def setSeed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initLogging(logFilename):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(message)s',
                        datefmt='%y-%m-%d %H:%M', filename=logFilename, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(message)s'))
    logging.getLogger('').addHandler(console)


def compute_metrics(all_results, top_k):
    if isinstance(top_k, int):
        top_k = [top_k]
    hr_result, ndcg_result = [], []
    for k in top_k:
        hr_list, ndcg_list = [], []
        for targets, preds in all_results:
            targets, preds = np.array(targets), np.array(preds)
            pred_sort_idx = np.argsort(preds)[::-1]
            top_k_idx = pred_sort_idx[:k]
            relevant_in_top_k = np.sum(targets[top_k_idx])
            hr_list.append(1.0 if relevant_in_top_k > 0 else 0.0)
            dcg = sum(targets[idx] / np.log2(i + 2) for i, idx in enumerate(top_k_idx))
            ideal_targets = np.sort(targets)[::-1]
            idcg = sum(ideal_targets[i] / np.log2(i + 2) for i in range(min(k, len(ideal_targets))))
            ndcg_list.append(dcg / idcg if idcg > 0.0 else 0.0)
        hr_result.append(np.mean(hr_list))
        ndcg_result.append(np.mean(ndcg_list))
    return hr_result, ndcg_result


def ldp_perturb_laplace(t: torch.Tensor, *, laplace_lambda: float) -> torch.Tensor:
    """
    Paper-style perturbation:
        theta <- theta + Laplace(0, lambda)

    Here lambda is the Laplace distribution scale (noise strength).
    lambda = 0 means adding all-zero noise (no perturbation).
    """
    if laplace_lambda is None:
        raise ValueError('laplace_lambda must be set')
    if laplace_lambda < 0:
        raise ValueError('laplace_lambda must be >= 0')

    x = t.detach()
    b = float(laplace_lambda)
    if b == 0.0:
        return x

    noise = torch.distributions.Laplace(
        loc=torch.zeros((), device=x.device, dtype=x.dtype),
        scale=torch.tensor(b, device=x.device, dtype=x.dtype),
    ).sample(x.shape)
    return x + noise
