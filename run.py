import math, torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

# ── Config (must match training exactly) ──
class Config:
    vocab_size = 32000; n_layer = 12; n_head = 12
    n_embd = 768; block_size = 512; dropout = 0.0; bias = False

cfg = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model (copy-paste from Cell 1 exactly) ──
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head = cfg.n_head; self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_drop(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x)[:, [-1], :])

# ── Load ──
print("Loading model...")
tokenizer = Tokenizer.from_file("tokenizer.json")
cfg.vocab_size = tokenizer.get_vocab_size()
bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")

ckpt = torch.load("model.pt", map_location=device)
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Model ready ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

# ── Generate ──
@torch.no_grad()
def generate(prompt, max_new_tokens=150, temperature=0.8, top_k=40):
    ids = [bos_id] + tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= cfg.block_size else x[:, -cfg.block_size:]
        logits = model(x_cond)[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if next_id.item() == eos_id: break
        x = torch.cat((x, next_id), dim=1)
    return tokenizer.decode(x[0].tolist())

# ── Chat ──
print("Chat ready (type 'quit' to exit)\n")
while True:
    prompt = input("You: ").strip()
    if not prompt: continue
    if prompt.lower() in ("quit","exit"): break
    print(f"AI: {generate(prompt)}\n")
