import argparse
from broad.build import build_broad

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet-based", 
        action=argparse.BooleanOptionalAction,
        help="Whether or not to include imagenet-based splits."
    )
    parser.add_argument(
        "--coco-based", 
        action=argparse.BooleanOptionalAction,
        help="Whether or not to include coco-based splits."
    )
    parser.add_argument(
        "--partial_broad_cache_dir",
        help="Optional path where to cache 'ServiceNow/PartialBROAD'.",
    )
    parser.add_argument(
        "--imagenet_cache_dir",
        help="Optional path where to cache 'imagenet-1k'.",
    )
    parser.add_argument(
        "--coco_cache_dir",
        help="Optional path where to cache 'HuggingFaceM4/COCO'.",
    )
    parser.add_argument(
        "--output_dir",
        help="Optional path where to save the BROAD processed dataset.",
    )
    parser.add_argument(
        "--hub_push_path",
        help="Optional hugging face datasets where to push the BROAD processed dataset.",
    )

    args = parser.parse_args()

    broad = build_broad(
        imagenet_based=args.imagenet_based,
        coco_based=args.coco_based,
        partial_broad_cache_dir=args.partial_broad_cache_dir,
        imagenet_cache_dir=args.imagenet_cache_dir, 
        coco_cache_dir=args.coco_cache_dir
    )

    if args.output_dir is not None:
        broad.save_to_disk(args.output_dir)
    
    if args.hub_push_path is not None:
        broad.push_to_hub(args.hub_push_path)