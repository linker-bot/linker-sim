"""Deterministic XML serialization for composed URDF/MJCF.

Goal: `git diff` only shows meaningful changes. Re-running the composer on
unchanged inputs produces byte-identical output.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from io import StringIO

# Consistent float formatting for xyz/rpy/scalar attributes. 9 sig figs is
# more than enough for URDF geometry while still producing short strings.
FLOAT_FMT = "{:.9g}"

# Attributes whose values are whitespace-separated float lists.
VECTOR_ATTRS = frozenset({"xyz", "rpy", "scale", "size", "axis"})


def fmt_float(x: float) -> str:
    return FLOAT_FMT.format(float(x))


def fmt_vec(vs) -> str:
    return " ".join(fmt_float(v) for v in vs)


def _normalize_vector_attrs(root: ET.Element) -> None:
    """Reformat known-vector attributes in-place for byte-stable output."""
    for elem in root.iter():
        for attr in list(elem.attrib):
            if attr in VECTOR_ATTRS:
                raw = elem.attrib[attr]
                parts = raw.strip().split()
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    continue
                elem.attrib[attr] = " ".join(fmt_float(v) for v in values)


def serialize(root: ET.Element, *, indent: str = "  ", xml_declaration: bool = True) -> str:
    """Serialize `root` to a deterministic string.

    - Pretty-prints with a consistent indent.
    - Normalizes float formatting in vector-valued attributes.
    - LF line endings, UTF-8, optional XML declaration.
    - A trailing newline is always appended.
    """
    # Copy so we don't mutate the caller's tree on repeat calls.
    clone = _deep_copy(root)
    _normalize_vector_attrs(clone)
    ET.indent(clone, space=indent, level=0)
    buf = StringIO()
    tree = ET.ElementTree(clone)
    tree.write(
        buf,
        encoding="unicode",
        xml_declaration=xml_declaration,
        short_empty_elements=True,
    )
    text = buf.getvalue()
    if xml_declaration:
        # ET emits `<?xml version='1.0' encoding='utf-8'?>` on Python 3.11;
        # normalize to the more common double-quoted UTF-8 form.
        text = text.replace(
            "<?xml version='1.0' encoding='utf-8'?>",
            '<?xml version="1.0" encoding="UTF-8"?>',
        )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text


def _deep_copy(elem: ET.Element) -> ET.Element:
    out = ET.Element(elem.tag, dict(elem.attrib))
    out.text = elem.text
    out.tail = elem.tail
    for child in elem:
        out.append(_deep_copy(child))
    return out
