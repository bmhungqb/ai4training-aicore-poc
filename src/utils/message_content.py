"""OpenAI-style chat message content parts for the VLM stages."""
from __future__ import annotations

import re
from string import Template
from typing import Iterable


def image_content(jpeg_b64: str) -> dict:
    """OpenAI-style image part for a base64 JPEG."""
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}


def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def labeled_frames(pairs: Iterable[tuple[str, str]]) -> list[dict]:
    """[(label, jpeg_b64), ...] -> alternating text/image content parts."""
    parts: list[dict] = []
    for label, jpeg_b64 in pairs:
        parts.append(text_content(label))
        parts.append(image_content(jpeg_b64))
    return parts


def render_template_content(template: str, subs: dict,
                            blocks: dict[str, list[dict]]) -> list[dict]:
    """Render a prompt template into a message-content list: each
    `$name` placeholder named in `blocks` is spliced with its pre-built
    content parts (frame labels + images), and the text around them gets
    `subs` substituted Template-style."""
    tokens = {f"${name}": parts for name, parts in blocks.items()}
    pattern = "(" + "|".join(re.escape(t) for t in tokens) + ")"
    content: list[dict] = []
    for part in re.split(pattern, template):
        if part in tokens:
            content.extend(tokens[part])
        elif part:
            content.append(text_content(Template(part).substitute(subs)))
    return content
