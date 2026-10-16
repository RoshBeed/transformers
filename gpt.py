import numpy as np

rng = np.random.default_rng()

# 1. Dimensions & Parameters
hp = {
    "seq_len": 8,
    "d_vocab": 40478,
    "d_model": 768,
    "h": 12,
    "N": 12,
    "max_positions": 512,
    "P_drop": 0.1,
    "eps": 1e-05,
    "training": False,
}
assert hp["d_model"] % hp["h"] == 0
hp["d_k"] = hp["d_v"] = hp["d_model"] // hp["h"]
hp["d_ff"] = 4 * hp["d_model"]

def init_weight(*shape):
    return rng.normal(0, 0.02, shape)

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
w["E_position"] = init_weight(hp["max_positions"], hp["d_model"])

w["W_Q_self"], w["W_K_self"], w["W_V_self"], w["W_O_self"] = init_multi_head_attention()
w["b_Q_self"] = np.zeros((hp["N"], hp["h"], hp["d_k"]))
w["b_K_self"] = np.zeros((hp["N"], hp["h"], hp["d_k"]))
w["b_V_self"] = np.zeros((hp["N"], hp["h"], hp["d_v"]))
w["b_O_self"] = np.zeros((hp["N"], hp["d_model"]))
w["gain_self"], w["bias_self"] = init_layer_norm()
w["W_1"], w["b_1"], w["W_2"], w["b_2"] = init_feed_forward_network()
w["gain_ffn"], w["bias_ffn"] = init_layer_norm()

# 2. Helpers
def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_scores = np.exp(x_shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

def dropout(x):
    if not hp["training"]:
        return x
    return x * rng.binomial(1, 1 - hp["P_drop"], x.shape) / (1 - hp["P_drop"])

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, sublayer, layer):
    gain, bias = w[f"gain_{sublayer}"][layer], w[f"bias_{sublayer}"][layer]
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + hp["eps"])
    return gain * x_norm + bias

def scaled_dot_product_attention(Q, K, V, mask):
    scores = (Q @ K.T) / np.sqrt(hp["d_k"])
    scores = np.where(mask, scores, -np.inf)
    return dropout(softmax(scores)) @ V

def multi_head_attention(x, layer, mask):
    heads = []
    for head in range(hp["h"]):
        Q = x @ w["W_Q_self"][layer, head] + w["b_Q_self"][layer, head]
        K = x @ w["W_K_self"][layer, head] + w["b_K_self"][layer, head]
        V = x @ w["W_V_self"][layer, head] + w["b_V_self"][layer, head]
        heads.append(scaled_dot_product_attention(Q, K, V, mask))
    return np.concatenate(heads, axis=-1) @ w["W_O_self"][layer] + w["b_O_self"][layer]

def feed_forward_network(x, layer):
    hidden = gelu(x @ w["W_1"][layer] + w["b_1"][layer])
    return hidden @ w["W_2"][layer] + w["b_2"][layer]

def add_and_norm(x, sublayer_output, layer, sublayer):
    x_residual = x + sublayer_output
    return layer_norm(x_residual, sublayer, layer)

def causal_mask(length):
    return np.tril(np.ones((length, length), dtype=bool))

def token_embedding(tokens):
    return w["E"][tokens]

def position_embedding(length):
    return w["E_position"][:length]

def linear(x):
    return x @ w["E"].T

# 3. Layers
def decoder_layer(x, layer):
    residual = x
    x = multi_head_attention(x, layer, causal_mask(len(x)))
    x = dropout(x)
    x = add_and_norm(residual, x, layer, "self")

    residual = x
    x = feed_forward_network(x, layer)
    x = dropout(x)
    x = add_and_norm(residual, x, layer, "ffn")
    return x

def decoder(x):
    for layer in range(hp["N"]):
        x = decoder_layer(x, layer)
    return x

def gpt(inputs):
    x = token_embedding(inputs) + position_embedding(len(inputs))
    y = decoder(x)
    logits = linear(y)
    return softmax(logits)

# 4. Forward Pass
inputs = rng.integers(0, hp["d_vocab"], size=hp["seq_len"])

# The original release applies dropout to the embedding tables before lookup.
w["E"], w["E_position"] = dropout(w["E"]), dropout(w["E_position"])
output_probabilities = gpt(inputs)

print(output_probabilities.shape)
