# transformer

My naive implementation of [Attention Is All You Need][paper] in numpy. Forward pass
only, one sequence at a time, no training.

It follows the equations rather than the optimised form: the `h` heads are `h`
separate projections concatenated before `W^O`, and `Q`, `K` and `V` each get their
own matmul, where an implementation built for speed would fuse them into single
matrices.

```shell
uv run transformer.py
```

[paper]: https://arxiv.org/abs/1706.03762

## The code

`transformer.py` is four sections: dimensions and parameters, helpers, layers,
forward pass. The dimensions are the base model's. Every sublayer is
`LayerNorm(x + Dropout(Sublayer(x)))`:

```python
residual, x = x, multi_head_attention(x, x, x, W_Q_self, W_K_self, W_V_self, W_O_self)
x = dropout(x)
x = add_and_norm(residual, x, gain_self, bias_self)
```

Hyperparameters and weights are plain variables, so `d_k` in the code is `d_k` in the
paper. Weights are indexed by layer: `W_Q_self_enc[layer]` holds that layer's `h`
query projections of `d_model × d_k`.

Dropout sits in three places, on one rate: the sum of the embeddings and the
positional encodings, the attention weights after the softmax, and each sublayer's
output before the residual add. It is the identity until `training` is set.
