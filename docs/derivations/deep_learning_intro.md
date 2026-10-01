# Deep learning for Terser: introduction (rows 8.1.1, 8.1.2, 8.2.2)

Code: [`models/neural/conv.py`](../../src/terser/models/neural/conv.py),
[`models/neural/textcnn.py`](../../src/terser/models/neural/textcnn.py) ·
Tests: `tests/test_conv_textcnn.py`

## 1. Representation learning

Every model in Phases 3–5 sees a prompt through features someone wrote: filler-lexicon hits,
hedge counts, constraint keywords, token-cost buckets, sentence position (row 3.1). The model
learns only how to *weight* those features. Representation learning moves the feature design into
training: a network maps raw input (tokens, or pretrained token embeddings) through layers of
learned transformations, and the intermediate activations *are* the features, chosen by gradient
descent to make the final loss small.

Why this should beat the Phase 3 hand features for **span importance** (which tokens can be
dropped without changing the answer):

- **Context decides importance.** "just" is filler in "just tell me" but meaningful in "just the
  first column". A lexicon feature fires on both; a convolution over neighbouring token embeddings
  sees the n-gram around the word.
- **Long tail of phrasings.** Hand features cover the hedges and fillers someone thought of.
  Embeddings place unseen paraphrases ("if it's not too much trouble") near seen ones, so a learned
  filter generalises across them.
- **Constraints are semantic.** Dropping "in Python 3.8" changes the answer; dropping "please"
  does not. Pretrained embeddings already encode that a version number is content and a courtesy is
  not; hand features approximate this with keyword lists.
- **Shared structure.** One convolution filter detects one pattern anywhere in the prompt (weight
  sharing), so the model needs far fewer parameters than a dense layer over all positions, and
  learns from every position of every prompt.

The cost: more data and compute, less interpretability, and a risk of learning dataset artefacts
(the synthetic-prompt circularity risk in the risk register). Learned models are therefore always
compared against the Phase 3/5 taggers on identical folds (row 8.2.4).

## 2. Decoder-only transformers (the generative rewriter, 8.3)

The rewriter is a decoder-only transformer (Qwen2.5-0.5B / 1.5B). In brief:

- **Tokens → vectors.** Text is split by a tokenizer into ids, each mapped to a learned embedding.
- **Self-attention.** Each position computes a query, key and value vector; its output is a
  weighted average of all values, weighted by softmax(q·k / √d). A **causal mask** lets position
  *t* attend only to positions ≤ *t*, so the model can be trained to predict the next token
  everywhere in a sequence at once. Positions are encoded by rotary embeddings (RoPE) in Qwen.
- **Blocks.** Attention (multi-head, here grouped-query) and a feed-forward MLP, each wrapped in a
  residual connection with RMSNorm, repeated *N* times (24 layers for 0.5B).
- **Output.** A final linear layer maps the last hidden state to logits over the vocabulary;
  training minimises next-token cross-entropy.
- **Generation.** At inference the model samples or greedily picks one token at a time, appending
  it to the input (a KV cache avoids recomputing past positions).

For Terser, supervised fine-tuning (SFT) trains on "casual prompt → oracle rewrite" pairs with the
loss masked to the rewrite tokens only, so the model learns to produce the rewrite conditioned on
the prompt. Decoding safeguards (8.3.7) and the quality gate (8.3.8) then decide whether its output
is used at all.

## 3. The convolution operation (row 8.1.2)

For input $x \in \mathbb R^{C_{in} \times L}$, kernel $w \in \mathbb R^{C_{out} \times C_{in}
\times k}$, bias $b \in \mathbb R^{C_{out}}$, zero padding $p$, stride $s$ and dilation $d$:

$$y_{o,t} = b_o + \sum_{c=1}^{C_{in}} \sum_{j=0}^{k-1} w_{o,c,j}\; x^{pad}_{c,\; ts + jd},
\qquad L_{out} = \Big\lfloor \frac{L + 2p - d(k-1) - 1}{s} \Big\rfloor + 1.$$

(This is cross-correlation; libraries call it convolution.) Worked example: $x = [1,2,3,4]$,
$w = [1, 0, -1]$, no padding: $y = [1-3, 2-4] = [-2, -2]$, and $L_{out} = (4 - 2 - 1) + 1 = 2$.
With $p = 1$ the padded input is $[0,1,2,3,4,0]$, giving $[-2,-2,-2,3]$ and $L_{out} = 4$
("same" padding, $p = (k-1)/2$ for odd $k$, stride 1). `conv1d` implements this and the test checks
it equals `torch.nn.Conv1d` for several stride / padding / dilation settings.

## 4. TextCNN parameter calculation (row 8.2.2)

A `Conv1d(C_in, C_out, k)` layer has one $C_{in} \times k$ kernel per output channel plus one
bias per output channel: $C_{in} \cdot k \cdot C_{out} + C_{out}$. A `Linear(m, n)` layer has
$m n + n$. Padding, ReLU, pooling and dropout add nothing. Token embeddings are computed upstream
(bge-small token states, d = 384) and are not part of the model.

| Layer | Formula | Params |
| :--- | :--- | ---: |
| Conv k = 3, 384 → 128 | 384 × 3 × 128 + 128 | 147,584 |
| Conv k = 5, 384 → 128 | 384 × 5 × 128 + 128 | 245,888 |
| Token head, 256 → 1 (both branches concatenated) | 256 × 1 + 1 | 257 |
| Prompt head, pooled 256 → 1 (max **or** avg pooling) | 256 × 1 + 1 | 257 |
| **Total** | | **393,986** |

With concatenated max **and** average pooling (row 8.2.3) the prompt head takes 512 inputs and has
512 + 1 = 513 parameters, for a total of 393,986 − 257 + 513 = 394,242.
`test_param_count_equals_numel` checks the closed form against `sum(p.numel())`.

## 5. Max vs average pooling (row 8.2.3, interpretation to be confirmed by the measured table)

Global **max** pooling keeps, per filter, the strongest response anywhere: the prompt head sees
"is there *any* droppable region". Global **average** pooling keeps the mean response: "*how much*
of the prompt is droppable", which grows with the share of filler. Concatenating both gives the
head both signals at the cost of 256 extra weights. Which one predicts compressibility better is an
empirical question answered by the 3-seed table in row 8.2.3, which needs the Phase 2 labels.
