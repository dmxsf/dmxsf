"""CLIP in miniature: the symmetric contrastive loss, a dual encoder and a toy image-caption dataset.

Course page: https://dmxsf.com/wiki/multimodal/multimodal-llms/

The data are 24x24 pictures of one coloured shape, with captions such as "a large red circle on the
left": 6 colours x 4 shapes x 2 sizes x 2 sides = 96 distinct captions. The images are rendered with numpy,
so the lab needs no dataset and no download. The point is the training objective, not the images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COLORS = {
    "red": (1.0, 0.1, 0.1),
    "green": (0.1, 0.9, 0.1),
    "blue": (0.15, 0.2, 1.0),
    "yellow": (1.0, 0.9, 0.1),
    "magenta": (1.0, 0.1, 0.9),
    "cyan": (0.1, 0.9, 0.9),
}
SHAPES = ("square", "circle", "triangle", "plus")
SIZES = ("small", "large")
SIDES = ("left", "right")
SLOTS = (tuple(COLORS), SHAPES, SIZES, SIDES)
SHAPE_OF_CLASS = tuple(len(s) for s in SLOTS)  # (6, 4, 2, 2)
N_CLASSES = math.prod(SHAPE_OF_CLASS)  # 96
N_WORDS = sum(len(s) for s in SLOTS)  # 14
CANVAS = 24


def class_words(classes: np.ndarray) -> np.ndarray:
    """Class ids -> a caption as four word ids (colour, shape, size, side), each slot with its own ids."""
    parts = np.unravel_index(np.asarray(classes), SHAPE_OF_CLASS)
    offsets = np.cumsum([0, *[len(s) for s in SLOTS[:-1]]])
    return np.stack([p + o for p, o in zip(parts, offsets, strict=True)], axis=-1)


def caption(cls: int) -> str:
    color, shape, size, side = (
        SLOTS[i][p] for i, p in enumerate(np.unravel_index(cls, SHAPE_OF_CLASS))
    )
    return f"a {size} {color} {shape} on the {side}"


def render(classes: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """One noisy picture per class id, shape ``(n, 3, 24, 24)``: the shape, its colour, size and side, with
    a jittered position and brightness, on a dark noisy background."""
    color, shape, size, side = np.unravel_index(np.asarray(classes), SHAPE_OF_CLASS)
    n = len(color)
    cx = np.where(side == 0, 7, 17) + rng.integers(-1, 2, n)
    cy = 12 + rng.integers(-2, 3, n)
    r = np.where(size == 0, 3, 6)
    ys, xs = np.mgrid[0:CANVAS, 0:CANVAS]
    dx, dy = xs[None] - cx[:, None, None], ys[None] - cy[:, None, None]
    rr = r[:, None, None]
    arm = np.maximum(1, rr // 3)
    masks = np.stack(
        [
            (np.abs(dx) <= rr) & (np.abs(dy) <= rr),  # square
            dx**2 + dy**2 <= rr**2,  # circle
            (dy >= -rr) & (dy <= rr) & (np.abs(dx) <= (dy + rr) / 2),  # triangle, apex up
            ((np.abs(dx) <= arm) & (np.abs(dy) <= rr)) | ((np.abs(dy) <= arm) & (np.abs(dx) <= rr)),
        ]
    )
    mask = masks[shape, np.arange(n)]
    rgb = np.array(list(COLORS.values()))[color] * rng.uniform(0.7, 1.0, n)[:, None]
    img = mask[:, None] * rgb[:, :, None, None] + rng.normal(0, 0.05, (n, 3, CANVAS, CANVAS))
    return torch.from_numpy(np.clip(img, 0, 1)).float()


# --- the loss ----------------------------------------------------------------------------------------


def clip_loss(a: torch.Tensor, b: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy over the image-text similarity matrix. ``a`` and ``b`` are ``(n, d)`` and
    unit length; ``scale`` is 1 / temperature. Row i asks "which text goes with image i", column i asks
    "which image goes with text i", and the matches sit on the diagonal."""
    logits = scale * a @ b.T
    target = torch.arange(len(logits))
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def clip_loss_by_hand(a: torch.Tensor, b: torch.Tensor, scale: float) -> float:
    """The formula of the paper, one sum at a time (slow; for checking ``clip_loss``)."""
    s = (scale * a @ b.T).double()
    n = len(s)
    total = 0.0
    for i in range(n):
        total += -math.log(math.exp(s[i, i]) / sum(math.exp(s[i, j]) for j in range(n)))
        total += -math.log(math.exp(s[i, i]) / sum(math.exp(s[j, i]) for j in range(n)))
    return total / (2 * n)


# --- the model -----------------------------------------------------------------------------------


class DualEncoder(nn.Module):
    """A small CNN for the image, a small MLP over the four caption words for the text, one shared
    embedding space, and the logit scale (1 / temperature) fixed or learned."""

    def __init__(self, d: int = 32, scale: float | None = None) -> None:
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, d),
        )
        self.words = nn.Embedding(N_WORDS, 16)
        self.text = nn.Sequential(nn.Linear(4 * 16, 64), nn.ReLU(), nn.Linear(64, d))
        start = math.log(1 / 0.07) if scale is None else math.log(scale)  # the paper's initial 0.07
        self.logit_scale = nn.Parameter(torch.tensor(start), requires_grad=scale is None)

    def scale(self) -> torch.Tensor:
        """1 / temperature, clipped at 100 as in the paper."""
        return self.logit_scale.exp().clamp(max=100.0)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image(x), dim=-1)

    def encode_text(self, words: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(self.words(words).flatten(1)), dim=-1)


@dataclass
class Result:
    accuracy: float  # zero-shot top-1 over all 96 captions on fresh pictures
    loss: float  # mean training loss over the last 20 steps
    scale: float  # 1 / temperature at the end
    history: list[float]


def zero_shot_accuracy(model: DualEncoder, per_class: int = 10, seed: int = 12345) -> float:
    """Classify fresh pictures by the nearest of the 96 caption embeddings: CLIP's zero-shot recipe."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(N_CLASSES), per_class)
    model.eval()
    with torch.no_grad():
        img = model.encode_image(render(labels, rng))
        txt = model.encode_text(torch.from_numpy(class_words(np.arange(N_CLASSES))))
    model.train()
    return float(((img @ txt.T).argmax(-1).numpy() == labels).mean())


def train(
    steps: int, batch: int, scale: float | None, seed: int, lr: float = 3e-3, d: int = 32
) -> tuple[DualEncoder, Result]:
    """Train from scratch with the symmetric loss. Each batch holds ``batch`` *different* captions (no two
    equal rows in the similarity matrix, so every off-diagonal entry really is a negative). ``scale=None``
    learns the temperature from the paper's initial 0.07; a number fixes 1 / temperature at that value."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DualEncoder(d, scale)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    history = []
    for _ in range(steps):
        classes = rng.choice(N_CLASSES, size=batch, replace=False)
        a = model.encode_image(render(classes, rng))
        b = model.encode_text(torch.from_numpy(class_words(classes)))
        loss = clip_loss(a, b, model.scale())
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    tail = history[-20:]
    return model, Result(
        zero_shot_accuracy(model), sum(tail) / len(tail), model.scale().item(), history
    )
