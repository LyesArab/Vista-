# ── re-export symbols from the top-level vista/utils.py (shadowed by this pkg)
import importlib.util as _ilu
import pathlib as _pl

_legacy_path = _pl.Path(__file__).parent.parent / "utils.py"
_spec = _ilu.spec_from_file_location("_vista_utils_legacy", _legacy_path)
_legacy = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

IGNORE_CATEGORIES  = _legacy.IGNORE_CATEGORIES
log                = _legacy.log
image_to_base64    = _legacy.image_to_base64
resize_image       = _legacy.resize_image
get_emergency_level = _legacy.get_emergency_level
set_seed           = _legacy.set_seed

del _ilu, _pl, _legacy_path, _spec, _legacy   # keep namespace clean

# ── rest of the package ───────────────────────────────────────────────────────
from .utils import *
from .grid import make_grid, linearize, delinearize, linearized_to_string
from .optuna import Optunizer
from .severity import (
    peak_severity,
    vehicle_severity_prompt,
    person_severity_prompt,
    emergency_vehicle_prompt,
    YOLO_INITIAL_CAPTION,
    YOLO_TO_CATEGORY,
    snap_to_vocabulary,
)