import time
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from tqdm import trange

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    causal: bool
    dropout: float = 0.0

    @property
    def head_size(self):
        return self.n_embd // self.n_head

# hyperparameters
# ------------

def gamma_func(ratio, mode):
    if mode == "linear":
        return 1 - ratio
    elif mode == "square":
        return 1 - ratio ** 2

torch.manual_seed(1337)

class Head(nn.Module):
    """ one head of self-attention """

    # def __init__(self, head_size, block_size, n_embd, causal=True):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.key = nn.Linear(self.config.n_embd, self.config.head_size, bias=False)
        self.query = nn.Linear(self.config.n_embd, self.config.head_size, bias=False)
        self.value = nn.Linear(self.config.n_embd, self.config.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size, self.config.block_size)))

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        if self.config.causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    # def __init__(self, num_heads, head_size, block_size, n_embd, causal=True):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([Head(self.config) for _ in range(self.config.n_head)])
        self.proj = nn.Linear(self.config.head_size * self.config.n_head, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    # def __init__(self, n_embd):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(self.config.n_embd, 4 * self.config.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.config.n_embd, self.config.n_embd),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    # def __init__(self, n_embd, n_head, block_size, causal=True):
    def __init__(self, config: GPTConfig):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.config = config
        head_size = self.config.n_embd // self.config.n_head
        self.sa = MultiHeadAttention(self.config)
        self.ffwd = FeedFoward(self.config)
        self.ln1 = nn.LayerNorm(self.config.n_embd)
        self.ln2 = nn.LayerNorm(self.config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    # def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, causal):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        # self.n_head = config.n_head
        # self.n_layer = config.n_layer
        # self.causal = config.causal

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd) # final layer norm
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx = rearrange(idx, 'B T S -> B (T S)')

        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

        pos_emb = self.position_embedding_table(torch.arange(T, device=self.position_embedding_table.weight.device))
        x = tok_emb + pos_emb # (B,T,C)

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, token_count, verbose=False):
        # idx is (B, T) array of indices in the current context
        if verbose:
            print(f"Generating {token_count} tokens with bs={idx.shape[0]}")
            st = time.time()

        for i in trange(token_count, desc='Generating tokens', disable=not verbose):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        if verbose:
            dur = time.time()-st
            print(f"Generation took {dur:.2f} seconds")
            print(f"tokens per second: {token_count/dur:.2f}")

        return idx

    def generate_maskgit(self, init, steps=1):
        self.eval()

        mask = torch.ones_like(init).bool()
        output = torch.zeros_like(init)+init # empty output (all mask tokens)
        unmasked_count = 0
        with torch.no_grad():
            for step in range(steps):
                logits, _ = self(output)
                probs = F.softmax(logits, dim=-1)
                B, T, C = probs.shape

                ratio = torch.tensor((step+1) / steps)
                gamma_r = gamma_func(ratio, mode='square')

                mask_count = (gamma_r * T).ceil().to(torch.int64)
                unmask_count = (256 - mask_count) - unmasked_count
                unmasked_count += unmask_count
                # print(f"{ratio=}, {gamma_r=}, {unmask_count=} {unmasked_count=}")

                # sample from the distribution
                samples_vec = torch.ones((B, T), device=probs.device, dtype=torch.int64)

                for i in range(B):
                    samples_vec[i, :] = torch.multinomial(probs[i, :], num_samples=1).view(-1)

                sampled_probs = probs[torch.arange(B).view(-1, 1), torch.arange(T).view(1, -1), samples_vec]
                sampled_probs[~mask] = 0. # mask out the already sampled tokens
                sorted_probs = torch.argsort(sampled_probs, dim=-1, descending=True)

                # unmask the top gamma_func(steps) tokens (for each example in the batch)
                top_k = sorted_probs[:, :unmask_count]
                mask[torch.arange(B).view(-1, 1), top_k] = 0
                output[:, top_k] = samples_vec[:, top_k]
                output[torch.arange(B).view(-1, 1), top_k] = samples_vec[torch.arange(B).view(-1, 1), top_k]
        self.train()

        return output



