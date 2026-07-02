"""
EventPool: the candidate/validated state machine.

Pure Python, no UI. Tracks two pools (candidates, validated), a current
position, and undo history for both validate and invalidate actions.
"""

import bisect


class EventPool:

    def __init__(self):
        self.candidates = []   # list[int] — event sample indices
        self.validated = []
        self.mode = "candidates"
        self.position = 0
        self._saved_positions = {"candidates": 0, "validated": 0}

        self.validate_undo = []     # entries: (idx_at_event, position_at_event)
        self.invalidate_undo = []   # entries: (idx, position, origin_mode)

    # ------------------------------------------------------------------
    # Loading & reset
    # ------------------------------------------------------------------
    def reset(self, candidates):
        """Replace candidates with a fresh list and clear all state."""
        self.candidates = list(candidates)
        self.validated.clear()
        self.mode = "candidates"
        self.position = 0
        self._saved_positions = {"candidates": 0, "validated": 0}
        self.validate_undo.clear()
        self.invalidate_undo.clear()

    # ------------------------------------------------------------------
    # Pool access
    # ------------------------------------------------------------------
    def current_pool(self):
        return self.candidates if self.mode == "candidates" else self.validated

    def current_index(self):
        """Sample index of the currently-selected event, or None if pool empty."""
        pool = self.current_pool()
        if not pool or self.position >= len(pool):
            return None
        return pool[self.position]

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def next(self):
        if self.position >= len(self.current_pool()) - 1:
            return False
        self.position += 1
        return True

    def prev(self):
        if self.position <= 0:
            return False
        self.position -= 1
        return True

    def toggle_mode(self):
        """Switch between candidates and validated views, preserving position."""
        self._saved_positions[self.mode] = self.position
        self.mode = "validated" if self.mode == "candidates" else "candidates"
        self.position = self._saved_positions[self.mode]
        self.position = min(self.position, max(0, len(self.current_pool()) - 1))

    # ------------------------------------------------------------------
    # Validate / invalidate
    # ------------------------------------------------------------------
    def validate(self):
        """In candidates mode: move current event to validated.
        In validated mode: move back to candidates."""
        idx = self.current_index()
        if idx is None:
            return False

        if self.mode == "candidates":
            self.validated.append(idx)
            self.validate_undo.append((idx, self.position))
            del self.candidates[self.position]
        else:
            insert_pos = bisect.bisect_left(self.candidates, idx)
            self.candidates.insert(insert_pos, idx)
            del self.validated[self.position]
            self.validate_undo = [
                (i, p) for (i, p) in self.validate_undo if i != idx
            ]

        self._clamp_position()
        return True

    def undo_validate(self):
        if not self.validate_undo:
            return False
        idx, _ = self.validate_undo.pop()
        if idx in self.validated:
            self.validated.remove(idx)
        bisect.insort(self.candidates, idx)
        if self.mode == "candidates":
            self.position = self.candidates.index(idx)
        return True

    def invalidate(self):
        """Permanently drop the current event from whichever pool we're in."""
        idx = self.current_index()
        if idx is None:
            return False
        self.invalidate_undo.append((idx, self.position, self.mode))
        pool = self.current_pool()
        del pool[self.position]
        self._clamp_position()
        return True

    def undo_invalidate(self):
        if not self.invalidate_undo:
            return False
        idx, _, origin_mode = self.invalidate_undo.pop()
        target = self.candidates if origin_mode == "candidates" else self.validated
        bisect.insort(target, idx)
        if origin_mode == self.mode:
            self.position = target.index(idx)
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _clamp_position(self):
        pool = self.current_pool()
        if self.position >= len(pool):
            self.position = max(0, len(pool) - 1)