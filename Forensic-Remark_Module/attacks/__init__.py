from .registry import ATTACK_REGISTRY, register_attack, build_attack
from . import stargan   # registers StarGANAttack
from . import stargan_v1  # registers StarGANV1FixedAttack
from . import stargan2  # registers Attack-StarGAN adapter (stargan2)
from . import simswap   # registers SimSwapAttack
from . import arc2face  # registers Arc2FaceAttack
from . import diffswap  # registers DiffSwapAttack
from . import reface  # registers ReFaceAttack
from . import face_adapter  # registers FaceAdapterAttack
