"""从 project_code.demo 导入 config 单例，统一入口。"""
import sys
import os

# 将 project_code 加入路径，确保 demo.py 中的相对路径正常
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from project_code.demo import config  # noqa: E402

__all__ = ["config"]
