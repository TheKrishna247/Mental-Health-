"""NLP tests for PS-87."""

import sys
from pathlib import Path

# Allow ``nlp`` package imports when running pytest from repo root or nlp/.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Note: vaderSentiment may emit DeprecationWarning on Python 3.14+ because the
# library uses codecs.open internally. This is a third-party compatibility issue,
# not a PS-87 NLP defect. Do not suppress unless/until upstream fixes it.
