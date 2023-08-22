import os
import json
from typing import Optional
from datasets import load_dataset, DatasetDict
from broad.delta_processor import DeltaProcessor
from broad.subcoco import subcoco

_SPLITS_TO_DELTA_PROCESS = {
    "adversarial_autoattack_resnet",
    "adversarial_autoattack_vit",
    "adversarial_pgd_resnet",
    "adversarial_pgd_vit",
}

_SPLITS_TO_SUBCOCO = {
    "CoComageNet-mono",
    "CoComageNet"
}


def get_subcoco_id2label(split: str) -> dict[int, int]:
    json_file = os.path.join(os.path.dirname(__file__), split + ".json")
    with open(json_file, "rt") as f:
        return {
            cocoid: int(label) 
            for label, cocoids in json.load(f).items()
            for cocoid in cocoids
        }


def build_broad(
        *,
        imagenet_based: bool = True,
        coco_based: bool = True,
        partial_broad_cache_dir: Optional[str] = None, 
        imagenet_cache_dir: Optional[str] = None, 
        coco_cache_dir: Optional[str] = None
        ) -> DatasetDict:
    broad = load_dataset("ServiceNow/PartialBROAD", cache_dir=partial_broad_cache_dir)

    if imagenet_based:
        # Process imagenet deltas.
        imagenet = load_dataset(
            "imagenet-1k", 
            split="validation", 
            cache_dir=imagenet_cache_dir)
        for split_name in _SPLITS_TO_DELTA_PROCESS:
            delta_processor = DeltaProcessor(
                imagenet, 
                only_keep=broad[split_name]["original_filename"],
            )
            broad[split_name] = broad[split_name].map(delta_processor)
    else:
        # Remove unprocessed imagenet deltas (to reduce chances of confusion).
        for split_name in _SPLITS_TO_DELTA_PROCESS:
            del broad[split_name]
    
    if coco_based:
        # Filter the subset of coco forming our datasets.
        coco = load_dataset(
            "HuggingFaceM4/COCO", 
            split="train+validation+test", 
            cache_dir=coco_cache_dir
        )
        for split_name in _SPLITS_TO_SUBCOCO:
            id2label = get_subcoco_id2label(split_name)
            broad[split_name] = subcoco(coco, id2label)

    return broad

