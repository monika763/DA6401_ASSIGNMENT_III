import argparse
import json
import math
import os
import random
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import BOS_IDX, EOS_IDX, PAD_IDX, Multi30kDataset, Vocab, collate_batch
from lr_scheduler import build_scheduler
from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, self.vocab_size)
        target = target.reshape(-1)

        non_pad = target.ne(self.pad_idx)
        if non_pad.sum() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits[non_pad], dim=-1)
        target = target[non_pad]

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    wandb=None,
    log_grad_norms: bool = False,
    grad_log_steps: int = 1000,
    global_step: int = 0,
) -> tuple[float, float, float, int]:
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    confidence_sum = 0.0
    confidence_count = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = loss_fn(logits, tgt_output)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if log_grad_norms and global_step < grad_log_steps:
                grad_metrics = query_key_grad_norms(model)
                grad_metrics["step"] = global_step
                if wandb is not None:
                    wandb.log(grad_metrics)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            non_pad = tgt_output.ne(PAD_IDX)
            total_correct += predictions.eq(tgt_output).masked_select(non_pad).sum().item()

            probs = F.softmax(logits, dim=-1)
            gold_confidence = probs.gather(-1, tgt_output.unsqueeze(-1)).squeeze(-1)
            confidence_sum += gold_confidence.masked_select(non_pad).sum().item()
            confidence_count += non_pad.sum().item()

        token_count = tgt_output.ne(PAD_IDX).sum().item()
        total_loss += loss.item() * max(token_count, 1)
        total_tokens += token_count

    avg_loss = total_loss / max(total_tokens, 1)
    token_accuracy = total_correct / max(total_tokens, 1)
    prediction_confidence = confidence_sum / max(confidence_count, 1)
    return avg_loss, token_accuracy, prediction_confidence, global_step


def query_key_grad_norms(model: Transformer) -> dict[str, float]:
    q_norm_sq = 0.0
    k_norm_sq = 0.0

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().norm(2).item()
        if ".W_q." in name:
            q_norm_sq += grad_norm * grad_norm
        elif ".W_k." in name:
            k_norm_sq += grad_norm * grad_norm

    return {
        "grad_norm/query": math.sqrt(q_norm_sq),
        "grad_norm/key": math.sqrt(k_norm_sq),
    }


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)

    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, model.pad_idx).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        logits = model.fc_out(out[:, -1])
        next_token = logits.argmax(dim=-1)
        next_token = next_token.masked_fill(finished, end_symbol)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= next_token.eq(end_symbol)
        if finished.all():
            break

    return ys


def _tokens_from_ids(ids: list[int], vocab, skip_special: bool = True) -> list[str]:
    tokens = []
    for idx in ids:
        idx = int(idx)
        if skip_special and idx in {PAD_IDX, BOS_IDX, EOS_IDX}:
            continue
        if hasattr(vocab, "lookup_token"):
            token = vocab.lookup_token(idx)
        elif hasattr(vocab, "itos"):
            token = vocab.itos[idx]
        else:
            token = vocab[idx]
        tokens.append(token)
    return tokens


def encode_sentence(text: str, tokenizer, vocab: Vocab, max_len: int, device: str) -> torch.Tensor:
    token_ids = [BOS_IDX]
    token_ids += vocab.lookup_indices(tokenizer(text))
    token_ids += [EOS_IDX]
    token_ids = token_ids[:max_len]
    if token_ids[-1] != EOS_IDX:
        token_ids[-1] = EOS_IDX
    return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def log_encoder_attention_maps(
    wandb,
    model: Transformer,
    sentence: str,
    dataset: Multi30kDataset,
    device: str,
    max_len: int,
    step: int,
) -> None:
    assert dataset.src_vocab is not None

    model.eval()
    src = encode_sentence(
        sentence,
        dataset.src_tokenizer,
        dataset.src_vocab,
        max_len=max_len,
        device=device,
    )
    src_mask = make_src_mask(src, model.pad_idx)
    model.encode(src, src_mask)

    last_layer = model.encoder.layers[-1]
    attn = last_layer.self_attn.last_attn_weights
    if attn is None:
        return

    src_tokens = ["<bos>"] + dataset.src_tokenizer(sentence)
    src_tokens = src_tokens[: max_len - 1] + ["<eos>"]
    src_tokens = src_tokens[: attn.size(-1)]
    attn = attn[0].detach().cpu().numpy()

    for head_idx in range(attn.shape[0]):
        table = wandb.Table(
            data=[
                [src_tokens[row], src_tokens[col], float(attn[head_idx, row, col])]
                for row in range(len(src_tokens))
                for col in range(len(src_tokens))
            ],
            columns=["query_token", "key_token", "attention"],
        )
        wandb.log(
            {
                f"attention/encoder_last_layer_head_{head_idx}": wandb.plot_table(
                    "wandb/heatmap/v1",
                    table,
                    fields={
                        "x": "key_token",
                        "y": "query_token",
                        "value": "attention",
                    },
                ),
                "epoch": step,
            }
        )


def _corpus_bleu(references: list[list[str]], hypotheses: list[list[str]], max_n: int = 4) -> float:
    if not hypotheses:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = {}
            for i in range(max(len(ref) - n + 1, 0)):
                gram = tuple(ref[i : i + n])
                ref_counts[gram] = ref_counts.get(gram, 0) + 1

            hyp_counts = {}
            for i in range(max(len(hyp) - n + 1, 0)):
                gram = tuple(hyp[i : i + n])
                hyp_counts[gram] = hyp_counts.get(gram, 0) + 1

            matches += sum(min(count, ref_counts.get(gram, 0)) for gram, count in hyp_counts.items())
            total += sum(hyp_counts.values())

        precisions.append((matches + 1.0) / (total + 1.0))

    ref_len = sum(len(ref) for ref in references)
    hyp_len = sum(len(hyp) for hyp in hypotheses)
    brevity = 0.0 if hyp_len == 0 else min(1.0, math.exp(1.0 - ref_len / hyp_len))
    score = brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return 100.0 * score


@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    references = []
    hypotheses = []

    for src, tgt in test_dataloader:
        src = src.to(device)
        src_mask = make_src_mask(src, model.pad_idx)
        decoded = greedy_decode(
            model,
            src,
            src_mask,
            max_len=max_len,
            start_symbol=BOS_IDX,
            end_symbol=EOS_IDX,
            device=device,
        )

        for pred_ids, gold_ids in zip(decoded.cpu().tolist(), tgt.tolist()):
            hypotheses.append(_tokens_from_ids(pred_ids, tgt_vocab))
            references.append(_tokens_from_ids(gold_ids, tgt_vocab))

    return _corpus_bleu(references, hypotheses)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model.config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint.get("epoch", 0))


def export_vocab_json(vocab: Vocab, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.stoi, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 Transformer experiments")

    parser.add_argument("--project", default="da6401-a3")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--scheduler", "--lr-scheduler", dest="scheduler", default="noam", choices=["fixed", "noam"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--positional-encoding", default="sinusoidal", choices=["sinusoidal", "learned"])
    parser.add_argument(
        "--attention-scaling",
        default="scaled",
        choices=["scaled", "unscaled"],
        help="Use scaled dot-product attention or remove the 1/sqrt(d_k) factor for Task 2.2.",
    )

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--best-model-name", default="best_model.pth")
    parser.add_argument("--source-vocab-name", default="source_vocab.json")
    parser.add_argument("--target-vocab-name", default="target_vocab.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-grad-norms", action="store_true")
    parser.add_argument("--grad-log-steps", type=int, default=1000)
    parser.add_argument("--val-bleu-every", type=int, default=1)
    parser.add_argument("--log-attention", action="store_true")
    parser.add_argument("--attention-sentence", default="Ein Mann spielt Gitarre.")

    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace):
    train_data = Multi30kDataset(
        split="train",
        min_freq=args.min_freq,
        max_len=args.max_len,
        max_examples=args.max_train_examples,
    )
    val_data = Multi30kDataset(
        split="validation",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        max_len=args.max_len,
        max_examples=args.max_eval_examples,
    )
    test_data = Multi30kDataset(
        split="test",
        src_vocab=train_data.src_vocab,
        tgt_vocab=train_data.tgt_vocab,
        max_len=args.max_len,
        max_examples=args.max_eval_examples,
    )

    collate = partial(collate_batch, pad_idx=PAD_IDX)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    return train_data, train_loader, val_loader, test_loader


def run_training_experiment() -> None:
    args = parse_args()
    set_seed(args.seed)

    wandb = None
    if args.use_wandb:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(
            project=args.project,
            name=args.run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    train_data, train_loader, val_loader, test_loader = build_dataloaders(args)
    assert train_data.src_vocab is not None
    assert train_data.tgt_vocab is not None

    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=args.d_model,
        N=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        positional_encoding=args.positional_encoding,
        use_attention_scaling=args.attention_scaling == "scaled",
        pad_idx=PAD_IDX,
        load_pretrained=False,
    ).to(args.device)

    lr = args.lr
    if lr is None:
        lr = 1.0 if args.scheduler == "noam" else 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
    )
    loss_fn = LabelSmoothingLoss(len(train_data.tgt_vocab), PAD_IDX, args.label_smoothing)

    if args.resume:
        load_checkpoint(args.resume, model, optimizer, scheduler)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, args.best_model_name)
    export_vocab_json(train_data.src_vocab, os.path.join(args.checkpoint_dir, args.source_vocab_name))
    export_vocab_json(train_data.tgt_vocab, os.path.join(args.checkpoint_dir, args.target_vocab_name))

    best_val_loss = float("inf")
    global_step = 0

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_confidence, global_step = run_epoch(
                train_loader,
                model,
                loss_fn,
                optimizer,
                scheduler,
                epoch_num=epoch,
                is_train=True,
                device=args.device,
                wandb=wandb,
                log_grad_norms=args.log_grad_norms,
                grad_log_steps=args.grad_log_steps,
                global_step=global_step,
            )
            val_loss, val_acc, val_confidence, _ = run_epoch(
                val_loader,
                model,
                loss_fn,
                optimizer=None,
                scheduler=None,
                epoch_num=epoch,
                is_train=False,
                device=args.device,
                global_step=global_step,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            val_bleu = None
            if args.val_bleu_every > 0 and epoch % args.val_bleu_every == 0:
                val_bleu = evaluate_bleu(
                    model,
                    val_loader,
                    train_data.tgt_vocab,
                    device=args.device,
                    max_len=args.max_len,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, best_model_path)

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                os.path.join(args.checkpoint_dir, "last_checkpoint.pth"),
            )

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_token_accuracy": train_acc,
                "train_prediction_confidence": train_confidence,
                "val_loss": val_loss,
                "val_token_accuracy": val_acc,
                "val_prediction_confidence": val_confidence,
                "lr": current_lr,
                "best_val_loss": best_val_loss,
            }
            if val_bleu is not None:
                metrics["val_bleu"] = val_bleu
            print(metrics)
            if wandb is not None:
                wandb.log(metrics)
                if args.log_attention:
                    log_encoder_attention_maps(
                        wandb,
                        model,
                        args.attention_sentence,
                        train_data,
                        args.device,
                        args.max_len,
                        step=epoch,
                    )

    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model)

    test_bleu = evaluate_bleu(
        model,
        test_loader,
        train_data.tgt_vocab,
        device=args.device,
        max_len=args.max_len,
    )
    print({"test_bleu": test_bleu})

    if wandb is not None:
        wandb.log({"test_bleu": test_bleu})
        wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
