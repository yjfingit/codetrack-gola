#!/usr/bin/env python3
"""Fix: ``diagnosis`` lives on model.codetrack, not on the top-level model."""
import pathlib, hashlib
p = pathlib.Path("trackit/runner/training/default/__init__.py")
src = p.read_text(encoding="utf-8"); orig = src

old = '''    diagnosis = getattr(module, "diagnosis", None)
    if diagnosis is None:
        return False'''
new = '''    # ``diagnosis`` is owned by the CodeTrack sub-module (codetrack/codetrack.py:87
    # ``self.diagnosis = SyndromeDiagnosis(...)``), NOT by the top-level model.
    # Walk the chain explicitly instead of guessing.
    diagnosis = None
    for holder in (module, getattr(module, "codetrack", None),
                   getattr(getattr(module, "module", None), "codetrack", None),
                   getattr(module, "module", None)):
        if holder is None:
            continue
        cand = getattr(holder, "diagnosis", None)
        if cand is not None:
            diagnosis = cand
            break
    if diagnosis is None:
        return False'''
assert src.count(old) == 1, "anchor not unique"
src = src.replace(old, new)
assert src != orig
p.write_text(src, encoding="utf-8")
print("FIXED diagnosis path")
print("md5:", hashlib.md5(src.encode()).hexdigest())
