# DA6401 Assignment 3: Transformer for German to English Translation

This repository contains a PyTorch implementation of the Transformer architecture from
"Attention Is All You Need" for German to English translation on the Multi30k dataset.

## Links

W&B Public Report: `<paste your public W&B report link here>`

GitHub Repository: `<paste your GitHub repository link here>`

## Project Structure

```text
.
├── dataset.py
├── lr_scheduler.py
├── model.py
├── plot_attention_heads.py
├── train.py
├── README.md
└── INSTRUCTIONS.md
```

## Files

- `model.py`: Transformer model, masks, attention, positional encodings, and `Transformer.infer()`.
- `dataset.py`: Multi30k loading, spaCy tokenization, vocabulary construction, and padding collate function.
- `lr_scheduler.py`: Noam learning rate scheduler and fixed scheduler helper.
- `train.py`: Training, validation, BLEU evaluation, checkpointing, W&B logging, and experiment arguments.
- `plot_attention_heads.py`: Generates 8 last-encoder-layer attention head heatmaps for Task 2.3.

## Dependencies

Install these in Kaggle/Colab before training:

```bash
pip install datasets wandb gdown spacy
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Kaggle Setup

Upload these files to `/kaggle/working` or your Kaggle project folder:

```text
model.py
train.py
dataset.py
lr_scheduler.py
plot_attention_heads.py
README.md
```

Then change into the folder containing the files:

```python
%cd /kaggle/working/da6401_assignment_3
```

Check that the correct files are being used:

```bash
grep -n "NotImplementedError" train.py || true
grep -n "val_token_accuracy" train.py
grep -n "use_attention_scaling" model.py
```

## W&B Login

```python
import wandb
wandb.login()
```

All runs use:

```text
project = da6401-a3
```

## Task 2.1: Necessity of Noam Scheduler

Fixed learning rate:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_1_fixed_lr_1e4 --scheduler fixed --lr 0.0001 --epochs 10 --val-bleu-every 0 --checkpoint-dir checkpoints_2_1_fixed
```

Noam scheduler:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_1_noam_scheduler --scheduler noam --lr 1.0 --warmup-steps 4000 --epochs 10 --val-bleu-every 0 --checkpoint-dir checkpoints_2_1_noam
```

W&B plots:

| Plot | X-axis | Y-axis |
|---|---|---|
| train loss | `epoch` | `train_loss` |
| validation accuracy | `epoch` | `val_token_accuracy` |
| learning rate | `epoch` | `lr` |

## Task 2.2: Scaling Factor Ablation

With scaling factor:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_2_scaled_attention --attention-scaling scaled --log-grad-norms --grad-log-steps 1000 --epochs 10 --checkpoint-dir checkpoints_2_2_scaled
```

Without scaling factor:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_2_unscaled_attention --attention-scaling unscaled --log-grad-norms --grad-log-steps 1000 --epochs 10 --checkpoint-dir checkpoints_2_2_unscaled
```

W&B plots:

| Plot | X-axis | Y-axis |
|---|---|---|
| query gradient norm | `_step` or `step` | `grad_norm/query` |
| key gradient norm | `_step` or `step` | `grad_norm/key` |
| train loss | `epoch` | `train_loss` |
| validation accuracy | `epoch` | `val_token_accuracy` |
| test BLEU | run name | `test_bleu` |

## Task 2.3: Attention Rollout and Head Specialization

Train and log attention:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_3_attention_head_specialization --log-attention --attention-sentence "Eine gruppe von männern lädt auf einen lastwagen." --epochs 10 --val-bleu-every 0 --checkpoint-dir checkpoints_2_3_attention
```

Generate 8 attention heatmap images:

```bash
python plot_attention_heads.py --checkpoint checkpoints_2_3_attention/best_model.pth --src-vocab checkpoints_2_3_attention/source_vocab.json --tgt-vocab checkpoints_2_3_attention/target_vocab.json --sentence "Eine gruppe von männern lädt auf einen lastwagen." --output-dir attention_heatmaps_2_3
```

Display in Kaggle:

```python
from IPython.display import Image, display
import glob

for path in sorted(glob.glob("attention_heatmaps_2_3/*.png")):
    print(path)
    display(Image(filename=path))
```

Heatmap interpretation:

| Plot | X-axis | Y-axis | Color |
|---|---|---|---|
| `encoder_head_0.png` to `encoder_head_7.png` | source token being attended to | source token doing the attending | attention weight |

Use all 8 heatmaps in the W&B report.

## Task 2.4: Positional Encoding vs Learned Embeddings

Sinusoidal positional encoding:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_4_sinusoidal_positional_encoding --positional-encoding sinusoidal --epochs 10 --val-bleu-every 1 --checkpoint-dir checkpoints_2_4_sinusoidal
```

Learned positional encoding:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_4_learned_positional_encoding --positional-encoding learned --epochs 10 --val-bleu-every 1 --checkpoint-dir checkpoints_2_4_learned
```

W&B plots:

| Plot | X-axis | Y-axis |
|---|---|---|
| validation accuracy | `epoch` | `val_token_accuracy` |
| validation loss | `epoch` | `val_loss` |
| validation BLEU | `epoch` | `val_bleu` |
| test BLEU | run name | `test_bleu` |

## Task 2.5: Decoder Sensitivity to Label Smoothing

Label smoothing with epsilon 0.1:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_5_label_smoothing_0_1 --label-smoothing 0.1 --epochs 10 --val-bleu-every 0 --checkpoint-dir checkpoints_2_5_smooth
```

Standard cross entropy with epsilon 0.0:

```bash
python train.py --use-wandb --project da6401-a3 --run-name 2_5_label_smoothing_0_0 --label-smoothing 0.0 --epochs 10 --val-bleu-every 0 --checkpoint-dir checkpoints_2_5_no_smooth
```

W&B plots:

| Plot | X-axis | Y-axis |
|---|---|---|
| train prediction confidence | `epoch` | `train_prediction_confidence` |
| validation prediction confidence | `epoch` | `val_prediction_confidence` |
| train loss, optional | `epoch` | `train_loss` |

## Final Best Model Training

Train the final model with the default/best settings:

```bash
python train.py --use-wandb --project da6401-a3 --run-name final_best_noam_scaled_sinusoidal_smooth --scheduler noam --lr 1.0 --warmup-steps 4000 --attention-scaling scaled --positional-encoding sinusoidal --label-smoothing 0.1 --epochs 10 --batch-size 64 --checkpoint-dir checkpoints_final
```

This produces:

```text
checkpoints_final/best_model.pth
checkpoints_final/source_vocab.json
checkpoints_final/target_vocab.json
```

## Google Drive Setup for Gradescope

Upload these files to Google Drive:

```text
best_model.pth
source_vocab.json
target_vocab.json
```

For each file:

1. Right click the file in Google Drive.
2. Click `Share`.
3. Set General access to `Anyone with the link`.
4. Set role to `Viewer`.
5. Copy the file link.
6. Extract the file ID.

Example:

```text
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

Use:

```text
FILE_ID
```

Update `model.py` inside `Transformer.__init__()`:

```python
src_vocab_url="https://drive.google.com/uc?id=SOURCE_VOCAB_FILE_ID",
tgt_vocab_url="https://drive.google.com/uc?id=TARGET_VOCAB_FILE_ID",
weights_url="https://drive.google.com/uc?id=BEST_MODEL_FILE_ID",
```

`model.py` downloads them as:

```python
src_vocab_path="source_vocab.json"
tgt_vocab_path="target_vocab.json"
weights_path="best_model.pth"
```

## Local Inference Test

Before submitting, test in a clean runtime:

```python
!rm -f source_vocab.json target_vocab.json best_model.pth

from model import Transformer

model = Transformer()
model.eval()
print(model.infer("Ein Mann spielt Gitarre."))
```

This should download all three files and print an English translation.

## Gradescope Submission

Submit code files:

```text
model.py
train.py
dataset.py
lr_scheduler.py
plot_attention_heads.py
README.md
```

Recommended: also submit the two small vocab files if Gradescope allows them:

```text
source_vocab.json
target_vocab.json
```

Do not submit:

```text
best_model.pth
```

The trained weights must be downloaded inside `Transformer.__init__()` using `gdown`.

## Autograder Contract

The autograder will call:

```python
from model import Transformer

model = Transformer().to(self.device)
model.eval()
english_sentence = model.infer(german_sentence)
```

Therefore:

- `Transformer()` must work without arguments.
- `Transformer.__init__()` must load vocabularies, tokenizer, architecture, and weights.
- `infer()` must accept one German sentence string.
- `infer()` must return one English sentence string.

## Notes

- Keep the Google Drive files available until marks are released.
- Keep the W&B report public during evaluation.
- Do not use test data for training or model selection.
