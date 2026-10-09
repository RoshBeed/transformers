import numpy as np

rng = np.random.default_rng()

# 1. Dimensions & Parameters
n = 8
d_vocab = 37000
d_model = 512
h = 8
N = 6
assert d_model % h == 0
d_k = d_v = d_model // h
d_ff = 4 * d_model
P_drop = 0.1
PE_base = 10000
eps = 1e-6
training = False

def init_weight(*shape):
    return rng.normal(0, shape[-2] ** -0.5, shape)

def init_multi_head_attention():
    return (init_weight(N, h, d_model, d_k), init_weight(N, h, d_model, d_k),
            init_weight(N, h, d_model, d_v), init_weight(N, h * d_v, d_model))

def init_ffn():
    return (init_weight(N, d_model, d_ff), np.zeros((N, d_ff)),
            init_weight(N, d_ff, d_model), np.zeros((N, d_model)))

def init_add_and_norm():
    return np.ones((N, d_model)), np.zeros((N, d_model))

E = init_weight(d_vocab, d_model)

W_Q_self_enc, W_K_self_enc, W_V_self_enc, W_O_self_enc = init_multi_head_attention()
gain_self_enc, bias_self_enc = init_add_and_norm()
W_1_enc, b_1_enc, W_2_enc, b_2_enc = init_ffn()
gain_ffn_enc, bias_ffn_enc = init_add_and_norm()

W_Q_self_dec, W_K_self_dec, W_V_self_dec, W_O_self_dec = init_multi_head_attention()
gain_self_dec, bias_self_dec = init_add_and_norm()
W_Q_cross_dec, W_K_cross_dec, W_V_cross_dec, W_O_cross_dec = init_multi_head_attention()
gain_cross_dec, bias_cross_dec = init_add_and_norm()
W_1_dec, b_1_dec, W_2_dec, b_2_dec = init_ffn()
gain_ffn_dec, bias_ffn_dec = init_add_and_norm()

# 2. Helpers
def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted), axis=axis, keepdims=True)

def dropout(x):
    if not training:
        return x
    return x * rng.binomial(1, 1 - P_drop, x.shape) / (1 - P_drop)

def layer_norm(x, gain, bias):
    x_norm = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + eps)
    return gain * x_norm + bias

def subsequent_mask(n):
    return np.tril(np.ones((n, n), dtype=bool))

def positional_encoding(n):
    pos, i = np.arange(n)[:, None], np.arange(d_model // 2)[None, :]
    angle = pos / PE_base ** (2 * i / d_model)
    PE = np.empty((n, d_model))
    PE[:, 0::2], PE[:, 1::2] = np.sin(angle), np.cos(angle)
    return PE

def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = (Q @ K.T) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return dropout(softmax(scores)) @ V

def multi_head_attention(Q, K, V, W_Q, W_K, W_V, W_O, mask=None):
    heads = [scaled_dot_product_attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i, mask)
             for W_Q_i, W_K_i, W_V_i in zip(W_Q, W_K, W_V)]
    return np.concatenate(heads, axis=-1) @ W_O

def ffn(x, W_1, b_1, W_2, b_2):
    return np.maximum(0, x @ W_1 + b_1) @ W_2 + b_2

def add_and_norm(x, sublayer_output, gain, bias):
    x_residual = x + sublayer_output
    return layer_norm(x_residual, gain, bias)

def embed(ids):
    return E[ids] * np.sqrt(d_model)

def output_probabilities(x):
    logits = x @ E.T
    return softmax(logits)

# 3. Layers
def encoder_layer(x, W_Q_self, W_K_self, W_V_self, W_O_self, gain_self, bias_self,
                  W_1, b_1, W_2, b_2, gain_ffn, bias_ffn):
    residual, x = x, multi_head_attention(x, x, x, W_Q_self, W_K_self, W_V_self, W_O_self)
    x = dropout(x)
    x = add_and_norm(residual, x, gain_self, bias_self)
    residual, x = x, ffn(x, W_1, b_1, W_2, b_2)
    x = dropout(x)
    x = add_and_norm(residual, x, gain_ffn, bias_ffn)
    return x

def decoder_layer(y, z, W_Q_self, W_K_self, W_V_self, W_O_self, gain_self, bias_self,
                  W_Q_cross, W_K_cross, W_V_cross, W_O_cross, gain_cross, bias_cross,
                  W_1, b_1, W_2, b_2, gain_ffn, bias_ffn):
    residual, y = y, multi_head_attention(y, y, y, W_Q_self, W_K_self, W_V_self, W_O_self,
                                          subsequent_mask(len(y)))
    y = dropout(y)
    y = add_and_norm(residual, y, gain_self, bias_self)
    residual, y = y, multi_head_attention(y, z, z, W_Q_cross, W_K_cross, W_V_cross, W_O_cross)
    y = dropout(y)
    y = add_and_norm(residual, y, gain_cross, bias_cross)
    residual, y = y, ffn(y, W_1, b_1, W_2, b_2)
    y = dropout(y)
    y = add_and_norm(residual, y, gain_ffn, bias_ffn)
    return y

def encoder(x):
    for layer in range(N):
        x = encoder_layer(x, W_Q_self_enc[layer], W_K_self_enc[layer], W_V_self_enc[layer],
                          W_O_self_enc[layer], gain_self_enc[layer], bias_self_enc[layer],
                          W_1_enc[layer], b_1_enc[layer], W_2_enc[layer], b_2_enc[layer],
                          gain_ffn_enc[layer], bias_ffn_enc[layer])
    return x

def decoder(y, z):
    for layer in range(N):
        y = decoder_layer(y, z, W_Q_self_dec[layer], W_K_self_dec[layer], W_V_self_dec[layer],
                          W_O_self_dec[layer], gain_self_dec[layer], bias_self_dec[layer],
                          W_Q_cross_dec[layer], W_K_cross_dec[layer], W_V_cross_dec[layer],
                          W_O_cross_dec[layer], gain_cross_dec[layer], bias_cross_dec[layer],
                          W_1_dec[layer], b_1_dec[layer], W_2_dec[layer], b_2_dec[layer],
                          gain_ffn_dec[layer], bias_ffn_dec[layer])
    return y

# 4. Forward Pass
ids_input = rng.integers(0, d_vocab, size=n)
ids_output = rng.integers(0, d_vocab, size=n)
PE = positional_encoding(n)

x = embed(ids_input) + PE
x = dropout(x)
z = encoder(x)

y = embed(ids_output) + PE
y = dropout(y)
y = decoder(y, z)
probabilities = output_probabilities(y)

print(z.shape, y.shape, probabilities.shape)
