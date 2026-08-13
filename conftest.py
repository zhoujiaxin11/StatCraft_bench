"""
conftest.py (repo root)
=======================
Keep ``python3 -m pytest -q`` at the repo root focused on the framework
suite. Task-level ``tasks/*/*/tests/test_outputs.py`` files are designed
to run inside a trial's workspace by ``framework/verifier_wrapper.py`` —
they resolve ``outcome.json`` / ``output/predictions.csv`` relative to
``Path('.')``, so collecting them here just guarantees two spurious
"file missing" failures on every clean checkout.

The exclusion is done via pytest's ``collect_ignore_glob`` hook rather
than a ``pytest.ini`` entry so an explicit
``pytest tasks/education_academia/159_gaokao_reform/tests/test_outputs.py``
(from inside a trial dir, which is how verifier_wrapper invokes it) still
works.
"""
from __future__ import annotations

collect_ignore_glob = [
    "tasks/*/*/tests/*",
]
