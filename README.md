# transformer

My NumPy implementations of [Transformer][paper] and [GPT][gpt-paper].

Each script initialises random weights and prints its output probability shapes:

```shell
uv run transformer.py
uv run gpt.py
```

**Transformer** predicts the next output token for sequence-to-sequence tasks
such as translation:

<img src="docs/figures/transformer.svg" alt="Transformer architecture with encoder, decoder, residual connections and positional encodings" width="707">

```python
def transformer(inputs, outputs_shifted_right=np.array([hp["bos_id"]])):
    x = token_embedding(inputs) + positional_encoding(len(inputs))
    x = dropout(x)
    z = encoder(x)

    y = token_embedding(outputs_shifted_right) + positional_encoding(len(outputs_shifted_right))
    y = dropout(y)
    y = decoder(y, z)
    logits = linear(y)
    return softmax(logits)
```

**GPT** pre-training uses next-token prediction (causal language modelling):

<img src="docs/figures/gpt.svg" alt="GPT (2018) architecture with a causal decoder, residual connections and next-token prediction head" width="357">

```python
def gpt(inputs):
    x = token_embedding(inputs) + position_embedding(len(inputs))
    y = decoder(x)
    logits = linear(y)
    return softmax(logits)
```

[paper]: https://arxiv.org/abs/1706.03762
[gpt-paper]: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
