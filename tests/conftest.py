import sys
from pathlib import Path

# Чтобы `import api` / `import textutils` работали при запуске pytest из корня
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
