import torch
import torch.nn as nn


class Client(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.latent_dim = args.latent_dim
        self.lora_rank = getattr(args, 'lora_rank', 4)

        self.item_emb_global = nn.Embedding(args.num_items, self.latent_dim)

        d, r = self.latent_dim, self.lora_rank
        self.lora_A = nn.Parameter(torch.zeros(d, r))
        self.lora_B = nn.Parameter(torch.zeros(r, d * 2))
        self.lora_bias = nn.Parameter(torch.zeros(d * 2))
        nn.init.normal_(self.lora_A, std=0.01)

        self.register_buffer('user_anchor', torch.zeros(self.latent_dim))

        self.affine_output = nn.Linear(self.latent_dim, 1)
        self.logistic = nn.Sigmoid()

    def set_user_anchor(self, anchor: torch.Tensor):
        self.user_anchor.copy_(anchor.detach().to(self.user_anchor.dtype))

    def forward(self, x):
        iid = x[:, 1]
        g_item = self.item_emb_global(iid)
        gap = g_item - self.user_anchor.unsqueeze(0)

        modulation = gap @ self.lora_A @ self.lora_B + self.lora_bias
        gamma, beta = torch.chunk(modulation, 2, dim=-1)
        gamma_act, beta_act = torch.tanh(gamma), torch.tanh(beta)

        p_item = g_item * (1.0 + gamma_act) + beta_act
        return self.logistic(self.affine_output(p_item))
