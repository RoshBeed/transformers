import numpy as np

rng = np.random.default_rng()

# 1. Dimensions & Parameters
hp = {
    "seq_len": 8,
    "d_vocab": 37000,
    "bos_id": 0,
    "d_model": 512,
    "h": 8,
    "N": 6,
    "P_drop": 0.1,
    "PE_base": 10000,
    "eps": 1e-06,
    "training": False,
}
assert hp["d_model"] % hp["h"] == 0
hp["d_k"] = hp["d_v"] = hp["d_model"] // hp["h"]
hp["d_ff"] = 4 * hp["d_model"]

def init_weight(*shape):
    return rng.normal(0, shape[-2] ** -0.5, shape)

def init_multi_head_attention():
    return (init_weight(hp["N"], hp["h"], hp["d_model"], hp["d_k"]),
            init_weight(hp["N"], hp["h"], hp["d_model"], hp["d_k"]),
            init_weight(hp["N"], hp["h"], hp["d_model"], hp["d_v"]),
            init_weight(hp["N"], hp["h"] * hp["d_v"], hp["d_model"]))

def init_feed_forward_network():
    return (init_weight(hp["N"], hp["d_model"], hp["d_ff"]), np.zeros((hp["N"], hp["d_ff"])),
            init_weight(hp["N"], hp["d_ff"], hp["d_model"]), np.zeros((hp["N"], hp["d_model"])))

def init_layer_norm():
    return np.ones((hp["N"], hp["d_model"])), np.zeros((hp["N"], hp["d_model"]))

w = {}
w["E"] = init_weight(hp["d_vocab"], hp["d_model"])

w["W_Q_self_enc"], w["W_K_self_enc"], w["W_V_self_enc"], w["W_O_self_enc"] = init_multi_head_attention()
w["gain_self_enc"], w["bias_self_enc"] = init_layer_norm()
w["W_1_enc"], w["b_1_enc"], w["W_2_enc"], w["b_2_enc"] = init_feed_forward_network()
w["gain_ffn_enc"], w["bias_ffn_enc"] = init_layer_norm()

w["W_Q_self_dec"], w["W_K_self_dec"], w["W_V_self_dec"], w["W_O_self_dec"] = init_multi_head_attention()
w["gain_self_dec"], w["bias_self_dec"] = init_layer_norm()
w["W_Q_cross_dec"], w["W_K_cross_dec"], w["W_V_cross_dec"], w["W_O_cross_dec"] = init_multi_head_attention()
w["gain_cross_dec"], w["bias_cross_dec"] = init_layer_norm()
w["W_1_dec"], w["b_1_dec"], w["W_2_dec"], w["b_2_dec"] = init_feed_forward_network()
w["gain_ffn_dec"], w["bias_ffn_dec"] = init_layer_norm()

# 2. Helpers
def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_scores = np.exp(x_shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

def dropout(x):
    if not hp["training"]:
        return x
    return x * rng.binomial(1, 1 - hp["P_drop"], x.shape) / (1 - hp["P_drop"])

def layer_norm(x, sublayer, layer):
    gain, bias = w[f"gain_{sublayer}"][layer], w[f"bias_{sublayer}"][layer]
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + hp["eps"])
    return gain * x_norm + bias

def causal_mask(length):
    return np.tril(np.ones((length, length), dtype=bool))

def positional_encoding(length):
    positions, i = np.arange(length)[:, None], np.arange(hp["d_model"] // 2)[None, :]
    angle = positions / hp["PE_base"] ** (2 * i / hp["d_model"])
    encoding = np.empty((length, hp["d_model"]))
    encoding[:, 0::2], encoding[:, 1::2] = np.sin(angle), np.cos(angle)
    return encoding

def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = (Q @ K.T) / np.sqrt(hp["d_k"])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return dropout(softmax(scores)) @ V

def multi_head_attention(Q, K, V, layer, sublayer, mask=None):
    heads = []
    for head in range(hp["h"]):
        Q_head = Q @ w[f"W_Q_{sublayer}"][layer, head]
        K_head = K @ w[f"W_K_{sublayer}"][layer, head]
        V_head = V @ w[f"W_V_{sublayer}"][layer, head]
        heads.append(scaled_dot_product_attention(Q_head, K_head, V_head, mask))
    return np.concatenate(heads, axis=-1) @ w[f"W_O_{sublayer}"][layer]

def feed_forward_network(x, layer, stack):
    hidden = np.maximum(0, x @ w[f"W_1_{stack}"][layer] + w[f"b_1_{stack}"][layer])
    return hidden @ w[f"W_2_{stack}"][layer] + w[f"b_2_{stack}"][layer]

def add_and_norm(x, sublayer_output, layer, sublayer):
    x_residual = x + sublayer_output
    return layer_norm(x_residual, sublayer, layer)

def token_embedding(tokens):
    return w["E"][tokens] * np.sqrt(hp["d_model"])

def linear(x):
    return x @ w["E"].T

# 3. Layers
def encoder_layer(x, layer):
    residual = x
    x = multi_head_attention(x, x, x, layer, "self_enc")
    x = dropout(x)
    x = add_and_norm(residual, x, layer, "self_enc")

    residual = x
    x = feed_forward_network(x, layer, "enc")
    x = dropout(x)
    x = add_and_norm(residual, x, layer, "ffn_enc")
    return x

def decoder_layer(y, z, layer):
    residual = y
    y = multi_head_attention(y, y, y, layer, "self_dec", causal_mask(len(y)))
    y = dropout(y)
    y = add_and_norm(residual, y, layer, "self_dec")

    residual = y
    y = multi_head_attention(y, z, z, layer, "cross_dec")
    y = dropout(y)
    y = add_and_norm(residual, y, layer, "cross_dec")

    residual = y
    y = feed_forward_network(y, layer, "dec")
    y = dropout(y)
    y = add_and_norm(residual, y, layer, "ffn_dec")
    return y

def encoder(x):
    for layer in range(hp["N"]):
        x = encoder_layer(x, layer)
    return x

def decoder(y, z):
    for layer in range(hp["N"]):
        y = decoder_layer(y, z, layer)
    return y

def transformer(inputs, outputs_shifted_right=np.array([hp["bos_id"]])):
    x = token_embedding(inputs) + positional_encoding(len(inputs))
    x = dropout(x)
    z = encoder(x)

    y = token_embedding(outputs_shifted_right) + positional_encoding(len(outputs_shifted_right))
    y = dropout(y)
    y = decoder(y, z)
    logits = linear(y)
    return softmax(logits)

# 4. Forward Pass
inputs = rng.integers(0, hp["d_vocab"], size=hp["seq_len"])
output_probabilities = transformer(inputs)

print(output_probabilities.shape)
