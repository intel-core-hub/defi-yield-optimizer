import sys
from pathlib import Path

# src/内のモジュールは互いを`from yields import ...`のようにトップレベル名で
# importし合う設計(パッケージ化されていない)ので、テストもsrc/をsys.pathに
# 加えた上で同じやり方でimportする。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
