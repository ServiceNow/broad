import numpy as np
from PIL import Image
from broad.utils.crypt import get_integrity_hash, encrypt_or_decrypt


class DeltaProcessor:
    """Alter an original_dataset by applying a delta/patch on it.
    
    This patch is applied by treating the original as an encryption key, and
    the provided delta is used as the ciphered message.

    This processor is it's own inverse, and may be used to obtain the delta.

    We assume that the `original_dataset` has the same structure as
    HuggingFace's imagenet-1k dataset.
    """
    def __init__(self, original_dataset, only_keep = None):
        if only_keep is not None:
            only_keep = frozenset(only_keep)

        # Directly access the `original_dataset` through its PyArrow interface.
        # Reason: We want the bytes of the original images to be exactly preserved
        #         across architectures/versions, and PIL cannot guarantee that.
        self._original_bytes = {}
        for batch in original_dataset.data.to_batches():
            for example in batch.to_pydict()["image"]:
                if only_keep is not None:
                    if example["path"] not in only_keep:
                        continue
                self._original_bytes[example["path"]] = example["bytes"]

    def __call__(self, example):
        original_bytes = self._original_bytes[example["original_filename"]]
        if example["original_hash"] == "none":
            example["original_hash"] = get_integrity_hash(original_bytes)
        
        delta = np.asarray(example["image"])
        delta_bytes = delta.tobytes(order="C")
        desired_bytes = encrypt_or_decrypt(
            input=delta_bytes, 
            key=original_bytes,
            key_hash=example["original_hash"])
        desired = np.frombuffer(desired_bytes, dtype="uint8").reshape(delta.shape)
        example["image"] = Image.fromarray(desired)

        return example
