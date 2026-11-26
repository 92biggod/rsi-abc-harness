"""A/B/C 三任务统一入口。

平台按 dataset_key 下发，一次评测会依次跑三个数据集：
  A → 六类气体识别（classification）
  B → 轴承剩余寿命（regression，分钟）
  C → C-MAPSS 剩余寿命（regression）

三个子模块各自带 `baselines` 包，同名会冲突，所以每次只把对应子目录放进
sys.path 首位，并清掉已加载的 `baselines.*`。一个进程只服务一个数据集，
因此这种隔离是充分的。
"""
import importlib.util
import sys
from pathlib import Path

TASKS = ("classification", "regression")

_ROOT = Path(__file__).resolve().parent
_CACHE = {}


def _log(message):
    print(f"[dispatch] {message}", file=sys.stderr, flush=True)


def _purge_baselines():
    for name in [key for key in sys.modules
                 if key == "baselines" or key.startswith("baselines.")]:
        del sys.modules[name]


def _load(key):
    key = str(key).upper()
    if key in _CACHE:
        return _CACHE[key]
    root = _ROOT / key.lower()
    module_path = root / "contest_submission.py"
    if not module_path.is_file():
        raise RuntimeError(f"no submission module for dataset {key}")
    _purge_baselines()
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(f"submission_{key.lower()}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _CACHE[key] = module
    _log(f"loaded module {key} from {root}")
    return module


def _keys(value, depth=0):
    if depth > 4:
        return set()
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found |= _keys(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value[:1]:
            found |= _keys(item, depth + 1)
    return found


def _dispatch_key(sample):
    key = getattr(sample, "dataset_key", None)
    if key:
        return str(key).upper()
    if getattr(sample, "task", None) == "classification":
        return "A"
    names = _keys(getattr(sample, "inputs", None))
    if "horizontal" in names or "vertical" in names:
        return "B"
    return "C"


def train(rnd):
    key = str(getattr(rnd, "dataset_key", "") or "").upper()
    if key not in ("A", "B", "C"):
        raise RuntimeError(f"unsupported dataset {key!r}")
    _log(f"train dataset={key} round={getattr(rnd, 'round_index', '?')}")
    return _load(key).train(rnd)


def predict(sample, model_dir):
    key = _dispatch_key(sample)
    _log(f"predict -> {key}")
    return _load(key).predict(sample, model_dir)
