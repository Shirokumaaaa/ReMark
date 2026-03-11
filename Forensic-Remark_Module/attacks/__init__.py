from .registry import ATTACK_REGISTRY, register_attack, build_attack
from . import stargan   # registers StarGANAttack
from . import simswap   # registers SimSwapAttack
from . import arc2face  # registers Arc2FaceAttack
from . import diffswap  # registers DiffSwapAttack
