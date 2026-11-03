#!/usr/bin/env python3
"""Fail on comments in Python files.

A Python file carries no comments: names, docstrings and the docs carry every
explanation. The check allows the ``# %%`` cell markers, a shebang, the
jupytext header, the lines of a jupytext markdown cell, and the ``noqa``,
``type:`` and ``pragma:`` directives of the tools.

Usage: ``python tools/check_comments.py FILE...``. The exit code is 1 when a
file has a comment.
"""

# %% imports ---------------------------------------------------------------------------
import io
import re
import sys
import tokenize
from pathlib import Path

# %% global variables ------------------------------------------------------------------
ALLOWED_COMMENT = re.compile(r"^#( %%|!|\s*(noqa|type:|pragma:))")


# %% private functions -----------------------------------------------------------------
def _prose_lines(lines: list[str]) -> set[int]:
    """Return the line numbers of the jupytext header and the markdown cells."""
    prose = set()
    in_header = bool(lines) and lines[0] == "# ---"
    in_markdown = False
    for number, line in enumerate(lines, 1):
        if in_header:
            prose.add(number)
            in_header = number == 1 or line != "# ---"
        elif line.startswith("# %%"):
            in_markdown = "[markdown]" in line
        elif in_markdown:
            prose.add(number)
    return prose


def _comments(path: str) -> list[tuple[int, str]]:
    """Return the line number and text of each forbidden comment in a file."""
    source = Path(path).read_text(encoding="utf-8")
    prose = _prose_lines(source.splitlines())
    return [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
        and token.start[0] not in prose
        and not ALLOWED_COMMENT.match(token.string)
    ]


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    findings = [
        f"{path}:{line}: {text}"
        for path in sys.argv[1:]
        for line, text in _comments(path)
    ]
    print("\n".join(findings), end="\n" if findings else "")
    sys.exit(bool(findings))
