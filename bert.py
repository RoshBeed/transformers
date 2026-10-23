import numpy as np

rng = np.random.default_rng()

# 1. Dimensions & Parameters
hp = {
    "seq_len": 8,
    "d_vocab": 30522,
    "d_model": 768,
    "h": 12,
    "N": 12,
    "max_positions": 512,
    "P_drop": 0.1,
    "eps": 1e-12,
    "training": False,
}
assert hp["d_model"] % hp["h"] == 0
hp["d_k"] = hp["d_v"] = hp["d_model"] // hp["h"]
hp["d_ff"] = 4 * hp["d_model"]

def init_weight(*shape):
    # BERT's normal initialisation is truncated at two standard deviations.
    weight = rng.normal(0, 0.02, shape)
    outside = np.abs(weight) > 0.04
    while outside.any():
        weight[outside] = rng.normal(0, 0.02, outside.sum())
        outside = np.abs(weight) > 0.04
    return weight

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
w["E_segment"] = init_weight(2, hp["d_model"])
w["gain_embed"], w["bias_embed"] = np.ones(hp["d_model"]), np.zeros(hp["d_model"])

w["W_Q_self"], w["W_K_self"], w["W_V_self"], w["W_O_self"] = init_multi_head_attention()
w["b_Q_self"] = np.zeros((hp["N"], hp["h"], hp["d_k"]))
w["b_K_self"] = np.zeros((hp["N"], hp["h"], hp["d_k"]))
w["b_V_self"] = np.zeros((hp["N"], hp["h"], hp["d_v"]))
w["b_O_self"] = np.zeros((hp["N"], hp["d_model"]))
w["gain_self"], w["bias_self"] = init_layer_norm()
w["W_1"], w["b_1"], w["W_2"], w["b_2"] = init_feed_forward_network()
w["gain_ffn"], w["bias_ffn"] = init_layer_norm()

w["W_mlm"], w["b_mlm"] = init_weight(hp["d_model"], hp["d_model"]), np.zeros(hp["d_model"])
w["gain_mlm"], w["bias_mlm"] = np.ones(hp["d_model"]), np.zeros(hp["d_model"])
w["b_vocab"] = np.zeros(hp["d_vocab"])
w["W_pool"], w["b_pool"] = init_weight(hp["d_model"], hp["d_model"]), np.zeros(hp["d_model"])
w["W_nsp"], w["b_nsp"] = init_weight(hp["d_model"], 2), np.zeros(2)

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

def layer_norm(x, sublayer, layer=None):
    gain, bias = w[f"gain_{sublayer}"], w[f"bias_{sublayer}"]
    if layer is not None:
        gain, bias = gain[layer], bias[layer]
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + hp["eps"])
    return gain * x_norm + bias

def scaled_dot_product_attention(Q, K, V):
    scores = (Q @ K.T) / np.sqrt(hp["d_k"])
    return dropout(softmax(scores)) @ V

def multi_head_attention(x, layer):
    heads = []
    for head in range(hp["h"]):
        Q = x @ w["W_Q_self"][layer, head] + w["b_Q_self"][layer, head]
        K = x @ w["W_K_self"][layer, head] + w["b_K_self"][layer, head]
        V = x @ w["W_V_self"][layer, head] + w["b_V_self"][layer, head]
        heads.append(scaled_dot_product_attention(Q, K, V))
    return np.concatenate(heads, axis=-1) @ w["W_O_self"][layer] + w["b_O_self"][layer]

def feed_forward_network(x, layer):
    hidden = gelu(x @ w["W_1"][layer] + w["b_1"][layer])
    return hidden @ w["W_2"][layer] + w["b_2"][layer]

def add_and_norm(x, sublayer_output, layer, sublayer):
    x_residual = x + sublayer_output
    return layer_norm(x_residual, sublayer, layer)

def token_embedding(tokens):
    return w["E"][tokens]

def position_embedding(length):
    return w["E_position"][:length]

def segment_embedding(segment_ids):
    return w["E_segment"][segment_ids]

def linear_gelu(x):
    return gelu(x @ w["W_mlm"] + w["b_mlm"])

def masked_lm_linear(x):
    return x @ w["E"].T + w["b_vocab"]

def linear_tanh(x):
    return np.tanh(x @ w["W_pool"] + w["b_pool"])

def next_sentence_linear(x):
    return x @ w["W_nsp"] + w["b_nsp"]

def masked_lm(x):
    x = linear_gelu(x)
    x = layer_norm(x, "mlm")
    logits = masked_lm_linear(x)
    return softmax(logits)

def next_sentence(x_cls):
    pooled = linear_tanh(x_cls)
    logits = next_sentence_linear(pooled)
    return softmax(logits)

# 3. Layers
def encoder_layer(x, layer):
    residual = x
    x = multi_head_attention(x, layer)
    x = dropout(x)
    x = add_and_norm(residual, x, layer, "self")

    residual = x
    x = feed_forward_network(x, layer)
    x = dropout(x)
    x = add_and_norm(residual, x, layer, "ffn")
    return x

def encoder(x):
    for layer in range(hp["N"]):
        x = encoder_layer(x, layer)
    return x

def bert(inputs, segment_ids, masked_positions):
    x = token_embedding(inputs) + position_embedding(len(inputs)) + segment_embedding(segment_ids)
    x = layer_norm(x, "embed")
    x = dropout(x)
    z = encoder(x)

    masked_probabilities = masked_lm(z[masked_positions])
    next_sentence_probabilities = next_sentence(z[0])
    return masked_probabilities, next_sentence_probabilities

# 4. Forward Pass
# [CLS] sentence A [SEP] sentence B [SEP]; 103 is [MASK].
inputs = np.array([101, 2009, 103, 102, 2023, 2003, 103, 102])
segment_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
masked_positions = np.array([2, 6])

masked_probabilities, next_sentence_probabilities = bert(
    inputs, segment_ids, masked_positions)

print(masked_probabilities.shape, next_sentence_probabilities.shape)
