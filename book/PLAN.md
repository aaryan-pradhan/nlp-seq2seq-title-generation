# Plan: Part "Extra revision" (chapters 08–15)

Source: an 8-video LLM playlist (1 h 52 min). Transcripts in `src/NN.txt` (auto-captions; 02 and 08 from frame sheets until their captions download), frames in `src/frames/NN-*.png`. `src/` is gitignored. The book never names the source (rule 5).

## Decisions
1. **Source**: transcripts plus frame contact sheets (one frame per 15 s, 3x3 tiles). Ideas and pictures are followed; every sentence and every drawing is ours.
2. **Purpose**: revision reference, complete: every idea in every video is covered. Scroll and recognise (L1), simple English, undergraduate maths (L24).
3. **Assumed knowledge**: chapters 03–07 (token, sequence-to-sequence model, encoder, decoder, Transformer, pretraining, fine-tuning, batch, epoch, learning rate, vocabulary, special tokens, padding, word embedding, cosine similarity, GloVe, teacher forcing, cross-entropy loss, greedy decoding, beam search, subword tokens, query, key, value, self-attention, cross-attention, instruction tuning, zero-shot). Revised in `recall` boxes, never re-defined. Plus vectors, matrices, exp, log, probability.
4. **Scope and order**: `\part{Extra revision}` after chapter 07, one chapter per video, playlist order. Nothing beyond the videos (L4).
5. **Numbers**: `tools/numbers.py` (asserts) recomputes every number; it is `VERIFY` in `book.mk`.

## Topic plan (learning DAG)
| Ch | Question | Video | Needs | Introduces | Cards | Status |
|---|---|---|---|---|---|---|
| 08 | How does a large language model write text? | 1 | token, decoder, Transformer, pretraining, fine-tuning, special tokens (recall) | large language model, decoder-only model, next-token prediction, autoregressive generation, logits, softmax function, end-of-sequence token, base model, supervised fine-tuning, prompt template, preference alignment, reward model | none: softmax is a row of the sampling card (ch 12) | todo |
| 09 | How is text made ready for training? | 2 | 08 | data cleaning (boilerplate, harmful, duplicate, low-quality text), data mixture, word-level and character-level tokenization, out-of-vocabulary word, byte pair encoding, token ID, context window, shifted labels | none: the windows count is a row of the training-steps card (ch 13) | todo |
| 10 | How do tokens become vectors? | 3 | 09, word embedding, cosine similarity (recall) | ordinal number problem, one-hot vector, bag of words, n-gram, semantic vs contextual meaning, vector analogy, continuous bag of words, skip-gram, embedding matrix, embedding layer, output layer, positional vector | Input vector of a token | todo |
| 11 | How does attention give each token its context? | 4 | 10, query, key, value, self-attention (recall) | attention score, scaled dot product, attention weight, causal mask, causal self-attention, quadratic cost, multi-head attention, output projection, layer normalization, residual connection, feed-forward network, decoder block, weight tying | Attention | todo |
| 12 | How is the next token picked? | 5 | 08, 11 | greedy sampling, temperature, top-k sampling, top-p sampling, renormalising | Sampling distribution | todo |
| 13 | How is a large language model pretrained? | 6 | 09, 11, 12, cross-entropy loss, learning rate (recall) | parameter count, training step, tokens trained on, training memory (weights, activations, gradients, optimizer state), data / pipeline / tensor parallelism, GELU, sparse categorical cross-entropy, random-guess loss, gradient descent, Adam optimizer, gradient accumulation, learning-rate warm-up, checkpoint | Parameter count · Training steps · Next-token loss · Training memory | todo |
| 14 | Why does generation keep a KV cache? | 7 | 11, 12 | repeated attention work, KV cache, latest-token query, linear vs quadratic cost, cache memory | none: unlocked in 15 | todo |
| 15 | How do MQA and GQA shrink the cache? | 8 | 14, multi-head attention | key-value head, multi-query attention, grouped-query attention | KV-cache memory (rows: MHA, GQA, MQA) | todo |
