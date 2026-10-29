"""Reply splitter: break long replies into Discord-sized chunks.

Discord caps messages at 2000 chars. Instead of hard-truncating, split
into at most max_parts chunks, preferring paragraph breaks, and never
leave a fenced code block open across a chunk boundary (close it at the
end of one chunk, reopen at the start of the next). Anything beyond the
cap is truncated with an ellipsis marker.

Pure functions (no Discord, no I/O) so they are unit-testable.
"""

import re

DISCORD_LIMIT = 2000
FENCE = "```"
ELLIPSIS = "…"

_BLANK_RE = re.compile(r"\n\s*\n")
_NEWLINE_RE = re.compile(r"\n")


def _inside_fence(text, open_at_start=False):
    """True if text ends inside an unclosed ``` block.

    open_at_start: the text continues a block reopened by the splitter.
    """
    return (text.count(FENCE) + int(bool(open_at_start))) % 2 == 1


CLOSER = "\n" + FENCE
REOPEN = FENCE + "\n"


def _avoid_fence_split(s, cut):
    """Move a hard cut back so it never lands inside a ``` marker."""
    p = s.find(FENCE, max(0, cut - 2), cut + 2)
    if p != -1 and p < cut:
        cut = p
    return max(1, cut)


def _find_cut(s, budget, open_at_start=False):
    """Index to cut s (len(s) > budget) so head (+ closer) fits budget.

    Prefers a blank line, then any newline, in the second half of the
    window; early cut points are used only when no later one exists.
    Cuts inside a fenced block are allowed as long as the "\n```"
    closer still fits. Falls back to a hard cut. Returns
    (cut, inside_fence) — inside_fence means the head must be closed
    and the next chunk reopened.
    """
    window = s[:budget]
    blanks = [m.start() for m in _BLANK_RE.finditer(window)]
    newlines = [m.start() for m in _NEWLINE_RE.finditer(window)]
    half = budget // 2
    for min_pos in (half, 0):
        for candidates in (blanks, newlines):
            for cut in reversed(candidates):
                if cut < min_pos:
                    break
                head = window[:cut].rstrip()
                if not head:
                    continue
                inside = _inside_fence(head, open_at_start)
                if inside and len(head) + len(CLOSER) > budget:
                    continue
                return len(head), inside
    cut = _avoid_fence_split(s, budget)
    inside = _inside_fence(s[:cut], open_at_start)
    if inside and cut + len(CLOSER) > budget:
        cut = _avoid_fence_split(s, budget - len(CLOSER))
        inside = _inside_fence(s[:cut], open_at_start)
    return cut, inside


def split_message(text, limit=DISCORD_LIMIT, max_parts=3):
    """Split text into <= max_parts chunks of <= limit chars.

    Fenced blocks are closed/reopened across boundaries, with room for
    the closer/reopener reserved up front so no content is dropped
    mid-reply. Only the tail past the cap is dropped, with an ellipsis
    marker. Never raises; always returns at least one chunk.
    """
    try:
        text = text if isinstance(text, str) else str(text)
    except Exception:
        return ["(unrenderable reply)"]
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DISCORD_LIMIT
    try:
        max_parts = int(max_parts)
    except (TypeError, ValueError):
        max_parts = 3
    limit = max(limit, len(FENCE) * 2 + 8)
    max_parts = max(max_parts, 1)

    if len(text) <= limit:
        return [text]

    parts, rest, carry = [], text, False
    for i in range(max_parts):
        prefix = REOPEN if carry else ""
        budget = limit - len(prefix)
        open_end = _inside_fence(rest, carry)
        if len(rest) + (len(CLOSER) if open_end else 0) <= budget:
            parts.append(prefix + rest + (CLOSER if open_end else ""))
            break
        if i == max_parts - 1:
            # Cap reached: truncate the tail with a marker, still closing
            # any open block so formatting never bleeds.
            room = budget - len(ELLIPSIS)
            head = rest[:_avoid_fence_split(rest, room)]
            if _inside_fence(head, carry):
                head = rest[:_avoid_fence_split(rest, room - len(CLOSER))]
            close = CLOSER if _inside_fence(head, carry) else ""
            parts.append(prefix + head + ELLIPSIS + close)
            break
        cut, inside = _find_cut(rest, budget, carry)
        parts.append(prefix + rest[:cut] + (CLOSER if inside else ""))
        rest = rest[cut:].lstrip("\n")
        carry = inside
        if not rest:
            break
    return parts or [text[:limit]]
