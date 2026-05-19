import math
import copy
import json
import os
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# SCALED DOT PRODUCT ATTENTION
# ==========================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:

    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ==========================================================
# MASKS
# ==========================================================

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    mask = (src == pad_idx)

    return mask.unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    batch_size, tgt_len = tgt.shape

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device),
        diagonal=1
    ).bool()

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

    return pad_mask | causal_mask


# ==========================================================
# MULTI HEAD ATTENTION
# ==========================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_scaling = use_scaling
        self.last_attn_weights = None

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
            use_scaling=self.use_scaling,
        )
        self.last_attn_weights = attn_weights.detach()

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(batch_size, -1, self.d_model)

        output = self.W_o(attn_output)

        return output


# ==========================================================
# POSITIONAL ENCODING
# ==========================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len]

        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):

        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


# ==========================================================
# FEED FORWARD
# ==========================================================

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):

        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.linear2(
            self.dropout(
                F.relu(self.linear1(x))
            )
        )


# ==========================================================
# ENCODER LAYER
# ==========================================================

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scaling=True):

        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):

        attn_output = self.self_attn(x, x, x, src_mask)

        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)

        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ==========================================================
# DECODER LAYER
# ==========================================================

class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scaling=True):

        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):

        attn_output = self.self_attn(x, x, x, tgt_mask)

        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, memory, memory, src_mask)

        x = self.norm2(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)

        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ==========================================================
# ENCODER
# ==========================================================

class Encoder(nn.Module):

    def __init__(self, layer, N):

        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ==========================================================
# DECODER
# ==========================================================

class Decoder(nn.Module):

    def __init__(self, layer, N):

        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


# ==========================================================
# TRANSFORMER
# ==========================================================

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size=None,
        tgt_vocab_size=None,
        d_model=256,
        N=4,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        max_len=5000,
        positional_encoding="sinusoidal",
        pad_idx=1,
        use_attention_scaling=True,
        load_pretrained=True,
        download_assets=True,
        src_vocab_path="source_vocab.json",
        tgt_vocab_path="target_vocab.json",
        weights_path="best_model.pth",
        src_vocab_url="https://drive.google.com/file/d/1cB_6TEth1TjQsl2yiylRGTrvZXnj5FQ8/view?usp=sharing",
        tgt_vocab_url="https://drive.google.com/file/d/1AAfu4SYnbDUCrXVqioXXVkfmQrEkXNVj/view?usp=sharing",
        weights_url="https://drive.google.com/file/d/1gA8b5azjl6KHO3zg-SEpnNvA9XzU6B7u/view?usp=sharing",
    ):

        super().__init__()

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.pad_idx = pad_idx
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_idx2token = None
        self.tgt_idx2token = None
        self.src_tokenizer = self._load_src_tokenizer()

        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "max_len": max_len,
            "positional_encoding": positional_encoding,
            "pad_idx": pad_idx,
            "use_attention_scaling": use_attention_scaling,
            "load_pretrained": False,
        }

        if load_pretrained:
            if download_assets:
                self._download_if_missing(src_vocab_path, src_vocab_url)
                self._download_if_missing(tgt_vocab_path, tgt_vocab_url)

            self.src_vocab = self._load_token_to_idx_vocab(src_vocab_path)
            self.tgt_vocab = self._load_token_to_idx_vocab(tgt_vocab_path)
            self.src_idx2token = {int(v): k for k, v in self.src_vocab.items()}
            self.tgt_idx2token = {int(v): k for k, v in self.tgt_vocab.items()}
            src_vocab_size = len(self.src_vocab)
            tgt_vocab_size = len(self.tgt_vocab)

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError(
                "src_vocab_size and tgt_vocab_size are required when load_pretrained=False."
            )

        # --------------------------------------------------
        # Build model architecture
        # --------------------------------------------------

        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        if positional_encoding == "sinusoidal":
            self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_len)
        elif positional_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(d_model, dropout, max_len=max_len)
        else:
            raise ValueError("positional_encoding must be 'sinusoidal' or 'learned'.")

        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            use_scaling=use_attention_scaling,
        )

        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout,
            use_scaling=use_attention_scaling,
        )

        self.encoder = Encoder(encoder_layer, N)

        self.decoder = Decoder(decoder_layer, N)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        if load_pretrained:
            if download_assets:
                self._download_if_missing(weights_path, weights_url)

            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if "positional_encoding.pe" in state_dict:
                if state_dict["positional_encoding.pe"].shape[1] != max_len:
                    state_dict.pop("positional_encoding.pe")
            self.load_state_dict(state_dict, strict=False)
        self.to(self.device)

        if load_pretrained:
            self.eval()

    def _download_if_missing(self, path: str, url: str) -> None:

        if os.path.exists(path):
            return

        import gdown

        file_id = self._extract_drive_file_id(url)
        if file_id is not None:
            gdown.download(id=file_id, output=path, quiet=False)
        else:
            gdown.download(url, path, quiet=False, fuzzy=True)

    def _extract_drive_file_id(self, url: str) -> Optional[str]:

        patterns = [
            r"/file/d/([^/]+)",
            r"[?&]id=([^&]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        if re.fullmatch(r"[-\w]{20,}", url):
            return url

        return None

    def _load_token_to_idx_vocab(self, path: str) -> dict[str, int]:

        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        if isinstance(vocab, dict) and "itos" in vocab:
            return {token: idx for idx, token in enumerate(vocab["itos"])}

        return {str(k): int(v) for k, v in vocab.items()}

    def _load_src_tokenizer(self):

        try:
            import spacy

            nlp = spacy.load("de_core_news_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
            return lambda text: [token.text.lower() for token in nlp(text)]
        except Exception:
            return lambda text: re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

    # ------------------------------------------------------

    def encode(self, src, src_mask):

        src = self.src_embedding(src) * math.sqrt(self.d_model)

        src = self.positional_encoding(src)

        return self.encoder(src, src_mask)

    # ------------------------------------------------------

    def decode(self, memory, src_mask, tgt, tgt_mask):

        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        tgt = self.positional_encoding(tgt)

        return self.decoder(tgt, memory, src_mask, tgt_mask)

    # ------------------------------------------------------

    def forward(self, src, tgt):

        src_mask = make_src_mask(src, self.pad_idx)

        tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        memory = self.encode(src, src_mask)

        output = self.decode(memory, src_mask, tgt, tgt_mask)

        output = self.fc_out(output)

        return output

    # ------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        german_sentence: str,
        max_len: int = 100,
    ) -> str:

        """
        Accepts a German sentence string,
        returns the translated English sentence string.
        """

        if self.src_vocab is None or self.tgt_vocab is None or self.tgt_idx2token is None:
            raise RuntimeError("infer() requires loaded source and target vocabularies.")

        self.eval()

        # ── Tokenize German input ──────────────────────────

        tokens = self.src_tokenizer(german_sentence)

        # ── Numericalize ───────────────────────────────────

        unk_idx = self.src_vocab.get("<unk>", 0)
        bos_idx = self.src_vocab.get("<bos>", self.src_vocab.get("<sos>", 2))
        eos_idx = self.src_vocab.get("<eos>", 3)

        token_ids = (
            [bos_idx]
            + [self.src_vocab.get(t, unk_idx) for t in tokens]
            + [eos_idx]
        )

        src = torch.tensor(
            token_ids, dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # ── Encode ────────────────────────────────────────

        src_mask = make_src_mask(src, self.pad_idx)

        memory = self.encode(src, src_mask)

        # ── Autoregressive decode ─────────────────────────

        tgt_bos = self.tgt_vocab.get("<bos>", self.tgt_vocab.get("<sos>", 2))
        tgt_eos = self.tgt_vocab.get("<eos>", 3)

        ys = torch.tensor(
            [[tgt_bos]], dtype=torch.long
        ).to(self.device)

        for _ in range(max_len):

            tgt_mask = make_tgt_mask(ys, self.pad_idx)

            out = self.decode(memory, src_mask, ys, tgt_mask)

            out = self.fc_out(out)

            next_token = out[:, -1, :].argmax(dim=-1).item()

            if next_token == tgt_eos:
                break

            ys = torch.cat(
                [ys, torch.tensor([[next_token]], device=self.device)],
                dim=1
            )

        # ── Decode token IDs to words ──────────────────────

        generated_ids = ys[0].tolist()[1:]   # skip <bos>

        words = [
            self.tgt_idx2token[i]
            for i in generated_ids
            if i not in (tgt_bos, tgt_eos)
        ]

        return " ".join(words)
