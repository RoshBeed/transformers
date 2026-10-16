"""Draw the README diagrams with shared dimensions and spacing."""

from html import escape
from math import hypot
from pathlib import Path


OUT = Path(__file__).parent / "figures"
BOX_WIDTH = 160
BOX_HEIGHT = 28
TWO_LINE_HEIGHT = 44
HEIGHTS = {
    "attention": 48, "masked_attention": 64, "forward": TWO_LINE_HEIGHT,
    "embedding": TWO_LINE_HEIGHT, "norm": BOX_HEIGHT,
    "linear": BOX_HEIGHT, "softmax": BOX_HEIGHT,
}
GAP = 28
PADDING = 20
HEADER = 40
RESIDUAL_GUTTER = 36
SUM_RADIUS = 12
STROKE = 1.6
COLOURS = {
    "attention": "#ffe2bb", "masked_attention": "#ffe2bb",
    "norm": "#f2f4c1", "forward": "#c2e8f7",
    "embedding": "#fce0e1", "linear": "#dbdfef", "softmax": "#cce7cf",
}


class Diagram:
    def __init__(self, width, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#000"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="#fff"/>',
        ]

    def text(self, x, y, label, size=14, anchor="middle"):
        lines = label.split("\n")
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="#000">')
        for index, line in enumerate(lines):
            offset = -(len(lines) - 1) * 9 if index == 0 else 18
            self.parts.append(f'<tspan x="{x}" dy="{offset}">{escape(line)}</tspan>')
        self.parts.append("</text>")

    def box(self, x, y, label, kind):
        height = HEIGHTS[kind]
        self.parts.append(f'<rect x="{x-BOX_WIDTH/2:g}" y="{y}" width="{BOX_WIDTH}" height="{height}" rx="6" fill="{COLOURS[kind]}" stroke="#000" stroke-width="{STROKE}"/>')
        self.text(x, y + height / 2, label)

    def path(self, points, arrow=True):
        commands = [f"M {points[0][0]:g} {points[0][1]:g}"]
        for index in range(1, len(points) - 1):
            a, b, c = points[index-1:index+2]
            before = hypot(b[0]-a[0], b[1]-a[1])
            after = hypot(c[0]-b[0], c[1]-b[1])
            radius = min(4, before / 2, after / 2)
            entry = (b[0] + (a[0]-b[0])*radius/before, b[1] + (a[1]-b[1])*radius/before)
            leave = (b[0] + (c[0]-b[0])*radius/after, b[1] + (c[1]-b[1])*radius/after)
            commands.append(f"L {entry[0]:g} {entry[1]:g} Q {b[0]:g} {b[1]:g} {leave[0]:g} {leave[1]:g}")
        commands.append(f"L {points[-1][0]:g} {points[-1][1]:g}")
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<path d="{" ".join(commands)}" fill="none" stroke="#000" stroke-width="{STROKE}" stroke-linejoin="round"{marker}/>')

    def frame(self, x, top, bottom, title, count, side="right"):
        left = x - BOX_WIDTH / 2 - PADDING
        right = x + BOX_WIDTH / 2 + RESIDUAL_GUTTER + PADDING
        self.parts.append(f'<rect x="{left:g}" y="{top}" width="{right-left:g}" height="{bottom-top}" rx="12" fill="#f3f3f4" stroke="#000" stroke-width="{STROKE}"/>')
        self.text(left + 12, top + HEADER / 2, title, anchor="start")
        bracket = left - 24 if side == "left" else right + 24
        tick = 12 if side == "left" else -12
        self.parts.append(f'<path d="M {bracket+tick:g} {top} H {bracket:g} V {bottom} H {bracket+tick:g}" fill="none" stroke="#000" stroke-width="{STROKE}"/>')
        self.text(bracket - 24 if side == "left" else bracket + 24, (top + bottom) / 2, f"{count}×", 16)

    def residual(self, x, start, norm):
        side = x + BOX_WIDTH / 2 + RESIDUAL_GUTTER
        self.path([(x, start), (side, start), (side, norm + BOX_HEIGHT / 2), (x + BOX_WIDTH / 2, norm + BOX_HEIGHT / 2)])

    def plus(self, x, y):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{SUM_RADIUS}" fill="#fff" stroke="#000" stroke-width="{STROKE}"/>')
        self.path([(x-6, y), (x+6, y)], arrow=False)
        self.path([(x, y-6), (x, y+6)], arrow=False)

    def sine(self, x, y):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="#fff" stroke="#000" stroke-width="{STROKE}"/>')
        self.parts.append(f'<path d="M {x-12} {y} C {x-8} {y-14}, {x-4} {y-14}, {x} {y} S {x+8} {y+14}, {x+12} {y}" fill="none" stroke="#000" stroke-width="{STROKE}"/>')

    def save(self, name):
        OUT.mkdir(exist_ok=True)
        (OUT / f"{name}.svg").write_text("\n".join(self.parts + ["</svg>"]) + "\n")


def stack(d, x, attention, activation, count, title, causal=False, cross=False, bracket="right"):
    """Place each sublayer with the same gap and the same residual gutter."""
    attention_kind = "masked_attention" if causal else "attention"
    attention_height = HEIGHTS[attention_kind]
    norm1 = attention - GAP - BOX_HEIGHT
    if cross:
        cross_attention = norm1 - 2 * GAP - HEIGHTS["attention"]
        cross_norm = cross_attention - GAP - BOX_HEIGHT
        forward = cross_norm - GAP - TWO_LINE_HEIGHT
    else:
        forward = norm1 - GAP - TWO_LINE_HEIGHT
    norm2 = forward - GAP - BOX_HEIGHT
    top = norm2 - HEADER
    bottom = attention + attention_height + GAP
    d.frame(x, top, bottom, title, count, bracket)
    d.box(x, attention, "Masked\nMulti-Head\nAttention" if causal else "Multi-Head\nAttention", attention_kind)
    d.box(x, norm1, "Add & Norm", "norm")
    d.box(x, forward, f"Feed Forward\n({activation})", "forward")
    d.box(x, norm2, "Add & Norm", "norm")
    d.path([(x, attention), (x, norm1 + BOX_HEIGHT)])
    d.residual(x, attention + attention_height + GAP / 2, norm1)
    if cross:
        d.box(x, cross_attention, "Multi-Head\nAttention", "attention")
        d.box(x, cross_norm, "Add & Norm", "norm")
        bus = cross_attention + HEIGHTS["attention"] + GAP
        d.path([(x, norm1), (x, bus), (x + 56, bus), (x + 56, cross_attention + HEIGHTS["attention"])])
        d.path([(x, cross_attention), (x, cross_norm + BOX_HEIGHT)])
        d.path([(x, cross_norm), (x, forward + TWO_LINE_HEIGHT)])
        d.residual(x, norm1 - GAP / 2, cross_norm)
        d.residual(x, cross_norm - GAP / 2, norm2)
        d.text(x + 68, bus - GAP / 2, "Q", 12, "start")
    else:
        d.path([(x, norm1), (x, forward + TWO_LINE_HEIGHT)])
        d.residual(x, norm1 - GAP / 2, norm2)
    d.path([(x, forward), (x, norm2 + BOX_HEIGHT)])
    result = {"output": norm2, "top": top, "bottom": bottom, "attention_bottom": attention + attention_height}
    if cross:
        result["cross_bus"] = bus
        result["cross_bottom"] = cross_attention + HEIGHTS["attention"]
    return result


def vocabulary_head(d, x, stack_top, output, label):
    linear = stack_top - GAP - BOX_HEIGHT
    softmax = linear - GAP - BOX_HEIGHT
    d.box(x, linear, "Linear", "linear")
    d.box(x, softmax, "Softmax", "softmax")
    d.path([(x, output), (x, linear + BOX_HEIGHT)])
    d.path([(x, linear), (x, softmax + BOX_HEIGHT)])
    d.path([(x, softmax), (x, softmax - GAP)])
    d.text(x, softmax - 2 * GAP, label)


def token_input(d, x, sum_y, label="Token\nEmbedding", inputs="Inputs"):
    top = sum_y + SUM_RADIUS + GAP
    bottom = top + TWO_LINE_HEIGHT
    d.box(x, top, label, "embedding")
    d.path([(x, top), (x, sum_y + SUM_RADIUS)])
    d.path([(x, bottom + GAP), (x, bottom)])
    d.text(x, bottom + 2 * GAP, inputs)


def transformer():
    d = Diagram(808, 908, "Transformer (2017)",
                "Six encoder and six decoder layers, with residual addition followed by layer normalisation. The encoder supplies decoder cross-attention keys and values. Token embeddings are summed with sinusoidal positional encodings.")
    encoder = stack(d, 224, 608, "ReLU", 6, "Encoder", bracket="left")
    decoder = stack(d, 584, 592, "ReLU", 6, "Decoder", causal=True, cross=True)
    bus = decoder["cross_bus"]
    d.path([(224, encoder["output"]), (224, encoder["top"] - GAP), (422, encoder["top"] - GAP), (422, bus), (576, bus), (576, decoder["cross_bottom"])])
    d.path([(528, bus), (528, decoder["cross_bottom"])])
    d.text(408, 420, "K, V", 12, "end")
    for x, side, embedding, inputs in [(224, -1, "Input\nEmbedding", "Inputs"), (584, 1, "Output\nEmbedding", "Outputs\n(shifted right)")]:
        sum_y = encoder["bottom"] + GAP + SUM_RADIUS
        d.plus(x, sum_y)
        d.path([(x, sum_y - SUM_RADIUS), (x, encoder["attention_bottom"] if side == -1 else decoder["attention_bottom"])])
        position = x + side * (SUM_RADIUS + 2 * GAP + 16)
        d.sine(position, sum_y)
        d.path([(position - side * 16, sum_y), (x + side * SUM_RADIUS, sum_y)])
        d.text(position + side * (16 + GAP + 32), sum_y, "Positional\nEncoding", 12)
        token_input(d, x, sum_y, embedding, inputs)
    vocabulary_head(d, 584, decoder["top"], decoder["output"], "Next output-token\nprobabilities")
    d.save("transformer")




def gpt():
    d = Diagram(408, 748, "GPT (2018): pre-training",
                "Twelve causal decoder layers with residual addition followed by layer normalisation. Token and learned position embeddings are summed. A linear vocabulary projection and softmax predict the next token.")
    decoder = stack(d, 160, 432, "GELU", 12, "Decoder", causal=True)
    vocabulary_head(d, 160, decoder["top"], decoder["output"], "Next-token probabilities")
    sum_y = decoder["bottom"] + GAP + SUM_RADIUS
    d.plus(160, sum_y)
    d.path([(160, sum_y - SUM_RADIUS), (160, decoder["attention_bottom"])])
    d.box(308, sum_y - TWO_LINE_HEIGHT / 2, "Learned Position\nEmbedding", "embedding")
    d.path([(228, sum_y), (172, sum_y)])
    token_input(d, 160, sum_y)
    d.save("gpt")


if __name__ == "__main__":
    transformer()
    gpt()
