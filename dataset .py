from collections import Counter
from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


class Vocab:
    def __init__(self, tokens: Iterable[str], min_freq: int = 2) -> None:
        counter = Counter(tokens)
        self.itos = list(SPECIAL_TOKENS)

        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.itos:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def lookup_token(self, idx: int) -> str:
        return self.itos[int(idx)]

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self[token] for token in tokens]

    def state_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: dict) -> "Vocab":
        vocab = cls([], min_freq=1)
        vocab.itos = list(state["itos"])
        vocab.stoi = {token: idx for idx, token in enumerate(vocab.itos)}
        return vocab


def _load_spacy_tokenizer(model_name: str) -> Callable[[str], list[str]]:
    try:
        import spacy

        nlp = spacy.load(model_name, disable=["tagger", "parser", "ner", "lemmatizer"])
        return lambda text: [token.text.lower() for token in nlp(text)]
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is required. Install it with: "
            f"python -m spacy download {model_name}"
        ) from exc


def get_tokenizers() -> tuple[Callable[[str], list[str]], Callable[[str], list[str]]]:
    return _load_spacy_tokenizer("de_core_news_sm"), _load_spacy_tokenizer("en_core_web_sm")


def load_multi30k_split(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install Hugging Face datasets to load Multi30k: pip install datasets"
        ) from exc

    split_aliases = {"valid": "validation", "val": "validation"}
    hf_split = split_aliases.get(split, split)
    return load_dataset("bentrevett/multi30k", split=hf_split)


def _read_pair(example) -> tuple[str, str]:
    if "de" in example and "en" in example:
        return example["de"], example["en"]
    if "translation" in example:
        return example["translation"]["de"], example["translation"]["en"]
    raise KeyError("Expected Multi30k example to contain de/en or translation fields.")


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: Vocab | None = None,
        tgt_vocab: Vocab | None = None,
        min_freq: int = 2,
        max_len: int = 100,
        max_examples: int | None = None,
    ) -> None:
        self.split = split
        self.max_len = max_len
        self.src_tokenizer, self.tgt_tokenizer = get_tokenizers()

        raw_data = load_multi30k_split(split)
        if max_examples is not None:
            raw_data = raw_data.select(range(min(max_examples, len(raw_data))))

        self.raw_pairs = [_read_pair(example) for example in raw_data]

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        if self.src_vocab is None or self.tgt_vocab is None:
            self.build_vocab(min_freq=min_freq)

        self.examples = self.process_data()

    def build_vocab(self, min_freq: int = 2) -> tuple[Vocab, Vocab]:
        src_tokens = []
        tgt_tokens = []

        for src_text, tgt_text in self.raw_pairs:
            src_tokens.extend(self.src_tokenizer(src_text))
            tgt_tokens.extend(self.tgt_tokenizer(tgt_text))

        self.src_vocab = Vocab(src_tokens, min_freq=min_freq)
        self.tgt_vocab = Vocab(tgt_tokens, min_freq=min_freq)
        return self.src_vocab, self.tgt_vocab

    def process_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        examples = []
        assert self.src_vocab is not None
        assert self.tgt_vocab is not None

        for src_text, tgt_text in self.raw_pairs:
            src_ids = [BOS_IDX]
            src_ids += self.src_vocab.lookup_indices(self.src_tokenizer(src_text))
            src_ids += [EOS_IDX]

            tgt_ids = [BOS_IDX]
            tgt_ids += self.tgt_vocab.lookup_indices(self.tgt_tokenizer(tgt_text))
            tgt_ids += [EOS_IDX]

            if len(src_ids) <= self.max_len and len(tgt_ids) <= self.max_len:
                examples.append(
                    (
                        torch.tensor(src_ids, dtype=torch.long),
                        torch.tensor(tgt_ids, dtype=torch.long),
                    )
                )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_idx: int = PAD_IDX):
    src_batch, tgt_batch = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src, tgt
