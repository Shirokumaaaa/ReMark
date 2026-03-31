__all__ = [
    "LandmarkBitEncoder",
    "build_landmark_bit_encoder",
    "bits_from_image_path",
    "bits_from_image_paths",
]


def __getattr__(name):
    if name in __all__:
        from .landmark_bits import (
            LandmarkBitEncoder,
            build_landmark_bit_encoder,
            bits_from_image_path,
            bits_from_image_paths,
        )

        mapping = {
            "LandmarkBitEncoder": LandmarkBitEncoder,
            "build_landmark_bit_encoder": build_landmark_bit_encoder,
            "bits_from_image_path": bits_from_image_path,
            "bits_from_image_paths": bits_from_image_paths,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
