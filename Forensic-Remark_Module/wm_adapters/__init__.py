from .registry import WM_REGISTRY, register_wm, build_wm_adapter
from . import fin  # registers FINAdapter
from . import sepmark  # registers SepMarkAdapter
from . import maskwm  # registers MaskWMAdapter
from . import trustmark  # registers TrustMarkAdapter (Q only)
