# Seq2Seq Title Generation (CS60075 — Assignment 2)

Generating titles for Wikipedia articles from their body text using Sequence-to-Sequence
language models — RNN-based encoder-decoders and pretrained Transformers. The task is
framed as extreme document compression: given the full text of an article, produce a short
title that captures its content, and evaluate against the reference title with ROUGE.

Full assignment brief: [`docs/assignment_spec.pdf`](docs/assignment_spec.pdf).
Write-up of design decisions, bugs hit, and fixes: [`docs/report.pdf`](docs/report.pdf).

## Repository structure

```
notebooks/
  01_preprocessing.ipynb        Part A — data cleaning (spaCy lemmatization, stopword removal)
  02_rnn_seq2seq.ipynb          Part B — GRU-based Seq2Seq models (B1, B2)
  03_transformer_models.ipynb   Part C — T5 fine-tuning and Flan-T5 zero-shot prompting (C1, C2)
docs/
  assignment_spec.pdf           Original assignment brief
  report.pdf                    Submitted report with results and design rationale
```

The notebooks were developed and run on Kaggle, using its GPU runtime and dataset mount
(`/kaggle/input/...`).

## Dataset

~14k Wikipedia articles (`text`, `title` columns) split into train (~14k) and test (100)
CSVs, as provided in the assignment brief. Not included in this repo — see
`docs/assignment_spec.pdf` for the download link. 500 articles are held out from the
training set as a validation split.

## Approach

### Part A — Preprocessing (`01_preprocessing.ipynb`)
Cleans article text and titles with spaCy (`en_core_web_sm`): lowercasing, stopword
removal, punctuation stripping, and lemmatization, run in batches via `nlp.pipe` for
speed (batch processing cut preprocessing time from ~1 hour to ~16 minutes for ~14k
articles).

### Part B — RNN Seq2Seq (`02_rnn_seq2seq.ipynb`)
A GRU encoder-decoder built up in stages:

- **B1 — Basic model**: bidirectional GRU `EncoderRNN` + unidirectional GRU `DecoderRNN`
  with teacher forcing, trained with cross-entropy loss (padding ignored) and evaluated
  with ROUGE + word-level F1. Vocabulary restricted to tokens appearing in ≥1% of articles.
- **B2 — Improvements**, each layered on top of the previous:
  - **GloVe embeddings** (`glove.6B.300d`) to initialize and freeze the encoder embedding layer.
  - **Hierarchical encoder** (`HierEncoderRNN`): word-level GRU → sentence-level GRU.
  - **Dual-GRU decoder** (`Decoder2RNN`): two stacked GRU layers instead of one.
  - **Beam search** decoding (width 5) as an alternative to greedy decoding at inference time.

Training uses HuggingFace `Accelerate` for mixed-precision (fp16) training, which gave
roughly a 20% training-time reduction and simplified device management.

### Part C — Transformers (`03_transformer_models.ipynb`)
- **C1 — Fine-tuning**: `google-t5/t5-small` fine-tuned end-to-end with `Seq2SeqTrainer`
  for 3 epochs, using beam search (width 5) for generation.
- **C2 — Zero-shot prompting**: `google/flan-t5-base` and `google/flan-t5-large`, each
  given two different instruction prompts, with no fine-tuning — testing how well an
  instruction-tuned model generalizes to title generation out of the box.

## Results

ROUGE F-measures on the test set (100 articles). RNN variants in Part B are cumulative —
each row adds the listed component on top of the row above it.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| B1 — Basic RNN Seq2Seq | 0.2499 | 0.0976 | 0.2469 |
| B2 — + GloVe embeddings | 0.4985 | 0.2630 | 0.4985 |
| B2 — + Hierarchical encoder | 0.3254 | 0.1093 | 0.3254 |
| B2 — + Dual-GRU decoder | 0.3049 | 0.0820 | 0.3016 |
| B2 — + Beam search | 0.3262 | 0.0710 | 0.3262 |
| C1 — T5-small (fine-tuned) | 0.8626 | 0.6516 | 0.8626 |
| C2 — Flan-T5-base, prompt 1 (zero-shot) | 0.6783 | 0.4782 | 0.6808 |
| C2 — Flan-T5-base, prompt 2 (zero-shot) | 0.6325 | 0.4610 | 0.6317 |
| C2 — Flan-T5-large, prompt 1 (zero-shot) | 0.8221 | 0.6383 | 0.8207 |
| C2 — Flan-T5-large, prompt 2 (zero-shot) | 0.8728 | 0.6452 | 0.8745 |

**Takeaways** (see `docs/report.pdf` for full discussion):
- GloVe pretrained embeddings gave the single biggest lift to the basic RNN — roughly
  doubling ROUGE-1/L and nearly tripling ROUGE-2.
- Stacking the hierarchical encoder, dual-GRU decoder, and beam search on top of GloVe
  each *hurt* scores relative to GloVe alone — added model capacity didn't help on this
  dataset/training budget, and beam search only marginally recovered ROUGE-1/L without
  helping ROUGE-2.
- Fine-tuned T5-small substantially outperforms every RNN variant and even the much
  larger zero-shot Flan-T5-large, showing the value of task-specific fine-tuning over
  scale alone for this dataset size.
- Zero-shot Flan-T5-large is competitive with fine-tuned T5-small without any
  gradient updates, and prompt wording measurably shifts its scores (e.g. ROUGE-1
  0.822 → 0.873 between prompts).

## Course context

CS60075 (Natural Language Processing) — Assignment 2: Seq2Seq Language Models.
