from .registry import WM_REGISTRY, register_wm, build_wm_adapter
from . import fin  # registers FINAdapter
from . import sepmark  # registers SepMarkAdapter
from . import lampmark  # registers LampMarkAdapter
from . import lawa  # registers LaWaAdapter / aliases
from . import sleepermark  # registers SleeperMarkAdapter / aliases
from . import tagwm  # registers TAGWMAdapter / aliases
from . import maskwm  # registers MaskWMAdapter
from . import trustmark  # registers TrustMarkAdapter (Q only)
