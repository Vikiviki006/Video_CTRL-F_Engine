"""Repository-level pytest configuration for ``backend/``.

Exists for exactly one reason: ``backend/test_gpu.py`` is an exploratory
SigLIP-2 smoke script, not a test module. It contains no test functions - it
loads the ~1.5 GB checkpoint and ``print``s the embedding shapes - but because
of its ``test_`` prefix pytest collected it as a test module and executed the
whole thing at *collection* time.

Two concrete symptoms this fixes:

* ``cd backend && pytest`` reported "no tests ran" for that file after spending
  ~2 minutes loading the model into GPU memory.
* ``pytest backend`` from the repo root aborted collection entirely with
  ``FileNotFoundError: test.jpg``, because the script opens that file through a
  relative path and is only resolvable when the CWD is ``backend/``.

``collect_ignore`` is the standard mechanism for this. Renaming the file would
work too, but keeping its name avoids breaking anyone who references it, and the
file is left fully runnable::

    cd backend && python test_gpu.py

The real test configuration lives in ``backend/tests/conftest.py``.
"""

collect_ignore = ["test_gpu.py"]