"""Minimal parser for the subset of Lua used by the shipped Twilight Struggle database.

The files under ``TwilightStruggle_Data/StreamingAssets/Lua`` that hold the map and
card definitions are pure data: a sequence of ``g_table["Key"] = { ... }`` assignments.
They use table constructors, string / number / boolean / nil literals, ``..`` string
concatenation, unary minus, and line comments. Nothing else -- no functions, no
control flow -- so a small hand-written scanner is enough and avoids needing a Lua
runtime just to read the data.

A Lua table is modelled as :class:`LuaTable`, which keeps positional entries and
named fields separately (card definitions genuinely use both at once, e.g.
``event_effects = { usageconditions = {...}; {"Foo", 3} }``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["LuaTable", "LuaSyntaxError", "parse_assignments", "to_jsonable"]


class LuaSyntaxError(SyntaxError):
    """Raised when the input steps outside the supported Lua subset."""


@dataclass
class LuaTable:
    """A Lua table constructor: ordered positional entries plus named fields."""

    seq: list[Any] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #

_NAME_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_NAME_BODY = _NAME_START | set("0123456789")
_DIGITS = set("0123456789")
_PUNCT = set("{}[]=,;()+*/")
_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "\\": "\\",
    '"': '"',
    "'": "'",
    "\n": "\n",
}


@dataclass(frozen=True)
class Token:
    kind: str  # "name" | "string" | "number" | "punct" | "concat" | "minus" | "eof"
    value: Any
    line: int


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, n = 0, 1, len(src)

    while i < n:
        ch = src[i]

        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch in " \t\r":
            i += 1
            continue

        # Comments. Long-bracket comments are not used by these files, but
        # recognising them keeps the scanner honest if the data ever changes.
        if src.startswith("--", i):
            if src.startswith("--[[", i):
                end = src.find("]]", i + 4)
                if end == -1:
                    raise LuaSyntaxError(f"unterminated block comment at line {line}")
                line += src.count("\n", i, end)
                i = end + 2
            else:
                end = src.find("\n", i)
                i = n if end == -1 else end
            continue

        if ch in ('"', "'"):
            text, i, line = _scan_string(src, i, line)
            tokens.append(Token("string", text, line))
            continue

        if src.startswith("..", i):
            tokens.append(Token("concat", "..", line))
            i += 2
            continue

        if ch in _DIGITS or (ch == "." and i + 1 < n and src[i + 1] in _DIGITS):
            num, i = _scan_number(src, i, line)
            tokens.append(Token("number", num, line))
            continue

        if ch == "-":
            tokens.append(Token("minus", "-", line))
            i += 1
            continue

        if ch in _NAME_START:
            j = i + 1
            while j < n and src[j] in _NAME_BODY:
                j += 1
            tokens.append(Token("name", src[i:j], line))
            i = j
            continue

        if ch in _PUNCT:
            tokens.append(Token("punct", ch, line))
            i += 1
            continue

        raise LuaSyntaxError(f"unexpected character {ch!r} at line {line}")

    tokens.append(Token("eof", None, line))
    return tokens


def _scan_string(src: str, i: int, line: int) -> tuple[str, int, int]:
    quote = src[i]
    i += 1
    out: list[str] = []
    while True:
        if i >= len(src):
            raise LuaSyntaxError(f"unterminated string starting on line {line}")
        ch = src[i]
        if ch == "\\":
            nxt = src[i + 1]
            if nxt in _DIGITS:  # \ddd decimal escape
                j = i + 1
                while j < len(src) and j < i + 4 and src[j] in _DIGITS:
                    j += 1
                out.append(chr(int(src[i + 1 : j])))
                i = j
                continue
            if nxt not in _ESCAPES:
                raise LuaSyntaxError(f"unknown escape \\{nxt} on line {line}")
            if nxt == "\n":
                line += 1
            out.append(_ESCAPES[nxt])
            i += 2
            continue
        if ch == quote:
            return "".join(out), i + 1, line
        if ch == "\n":
            raise LuaSyntaxError(f"newline in string on line {line}")
        out.append(ch)
        i += 1


def _scan_number(src: str, i: int, line: int) -> tuple[int | float, int]:
    j = i
    seen_dot = False
    while j < len(src):
        if src[j] in _DIGITS:
            j += 1
        elif src[j] == "." and not seen_dot and not src.startswith("..", j):
            seen_dot = True
            j += 1
        else:
            break
    text = src[i:j]
    try:
        return (float(text) if seen_dot else int(text)), j
    except ValueError as exc:  # pragma: no cover - scanner shape prevents this
        raise LuaSyntaxError(f"bad number {text!r} on line {line}") from exc


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    @property
    def cur(self) -> Token:
        return self.tokens[self.pos]

    def take(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind: str, value: Any = None) -> Token:
        tok = self.take()
        if tok.kind != kind or (value is not None and tok.value != value):
            want = value if value is not None else kind
            raise LuaSyntaxError(
                f"expected {want!r} but found {tok.value!r} on line {tok.line}"
            )
        return tok

    def at_punct(self, value: str) -> bool:
        return self.cur.kind == "punct" and self.cur.value == value

    # -- values ----------------------------------------------------------- #
    #
    # Precedence, loosest first: `..` then `+ -` then `* /` then unary minus.
    # Arithmetic appears because several cards pack multiple numbers into one
    # effect argument, e.g. Korean War's `2 + (2 * 256)` meaning 2 military
    # operations and 2 victory points. Folding it here keeps the packed value
    # identical to what the game computes.

    def parse_value(self) -> Any:
        left = self._parse_additive()
        if self.cur.kind != "concat":
            return left
        parts = [left]
        while self.cur.kind == "concat":
            self.take()
            parts.append(self._parse_additive())
        return "".join(str(p) for p in parts)

    def _parse_additive(self) -> Any:
        left = self._parse_multiplicative()
        while self.cur.kind == "minus" or self.at_punct("+"):
            tok = self.take()
            right = self._parse_multiplicative()
            a, b = self._nums(left, right, tok)
            left = a - b if tok.kind == "minus" else a + b
        return left

    def _parse_multiplicative(self) -> Any:
        left = self._parse_unary()
        while self.at_punct("*") or self.at_punct("/"):
            tok = self.take()
            right = self._parse_unary()
            a, b = self._nums(left, right, tok)
            left = a * b if tok.value == "*" else a / b
        return left

    def _parse_unary(self) -> Any:
        if self.cur.kind == "minus":
            tok = self.take()
            operand = self._parse_unary()
            if not isinstance(operand, (int, float)) or isinstance(operand, bool):
                raise LuaSyntaxError(f"unary minus on non-number at line {tok.line}")
            return -operand
        return self._parse_primary()

    @staticmethod
    def _nums(left: Any, right: Any, tok: Token) -> tuple[int | float, int | float]:
        for operand in (left, right):
            if not isinstance(operand, (int, float)) or isinstance(operand, bool):
                raise LuaSyntaxError(
                    f"arithmetic on non-number {operand!r} at line {tok.line}"
                )
        return left, right

    def _parse_primary(self) -> Any:
        tok = self.cur

        if tok.kind == "number":
            self.take()
            return tok.value

        if tok.kind == "string":
            self.take()
            return tok.value

        if tok.kind == "name":
            self.take()
            if tok.value == "true":
                return True
            if tok.value == "false":
                return False
            if tok.value == "nil":
                return None
            raise LuaSyntaxError(f"unexpected identifier {tok.value!r} at line {tok.line}")

        if self.at_punct("{"):
            return self.parse_table()

        if self.at_punct("("):
            self.take()
            inner = self.parse_value()
            self.expect("punct", ")")
            return inner

        raise LuaSyntaxError(f"unexpected token {tok.value!r} on line {tok.line}")

    def parse_table(self) -> LuaTable:
        self.expect("punct", "{")
        table = LuaTable()

        while not self.at_punct("}"):
            # `[expr] = value`
            if self.at_punct("["):
                self.take()
                key = self.parse_value()
                self.expect("punct", "]")
                self.expect("punct", "=")
                table.fields[str(key)] = self.parse_value()

            # `name = value`
            elif self.cur.kind == "name" and self.tokens[self.pos + 1].kind == "punct" \
                    and self.tokens[self.pos + 1].value == "=":
                name = self.take().value
                self.take()  # '='
                table.fields[name] = self.parse_value()

            # positional entry
            else:
                table.seq.append(self.parse_value())

            if self.at_punct(",") or self.at_punct(";"):
                self.take()
            elif not self.at_punct("}"):
                raise LuaSyntaxError(
                    f"expected ',' ';' or '}}' but found {self.cur.value!r} "
                    f"on line {self.cur.line}"
                )

        self.expect("punct", "}")
        return table


def parse_assignments(src: str, table_name: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(key, value)`` for each ``<table_name>["key"] = value`` statement.

    Statements assigning to other globals are skipped, so a file may mix several
    databases without confusing the caller.
    """
    parser = _Parser(tokenize(src))

    while parser.cur.kind != "eof":
        target = parser.expect("name").value
        parser.expect("punct", "[")
        key = parser.expect("string").value
        parser.expect("punct", "]")
        parser.expect("punct", "=")
        value = parser.parse_value()
        if parser.at_punct(";"):
            parser.take()
        if target == table_name:
            yield key, value


def to_jsonable(value: Any) -> Any:
    """Convert parsed values to JSON-safe structures.

    A table with only positional entries becomes a list and one with only named
    fields becomes an object; a table using both keeps its positional entries
    under the reserved ``"_seq"`` key.
    """
    if isinstance(value, LuaTable):
        seq = [to_jsonable(v) for v in value.seq]
        fields = {k: to_jsonable(v) for k, v in value.fields.items()}
        if fields and seq:
            return {**fields, "_seq": seq}
        if fields:
            return fields
        return seq
    return value
