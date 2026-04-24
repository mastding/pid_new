"""Legacy data-understanding skills kept for compatibility.

Newer workflow-oriented skills now live in dedicated subpackages such as
``core.skills.window``. This package remains as a compatibility layer for
existing prompts and tests.
"""

from core.skills.data_understanding import detect_candidate_windows  # noqa: F401
from core.skills.data_understanding import load_dataset  # noqa: F401
from core.skills.data_understanding import summarize_data  # noqa: F401
