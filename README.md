# BROAD

This repo contains functionality to download data and prepare BROAD, an image evaluation dataset for broad OOD detection.

After installing requirements, the dataset can be built using the `build_broad.py` script. The following example will generate the entire dataset including the componenets that depend on imagenet and coco. This behaviour can be modified by dropping the ` --imagenet-based` or `--coco-based` options.

Note that outputs can be saved to disk and/or pushed to a private[^1] huggingface repo. Also, downloading datasets might require logging in to huggingface's hub beforehand (i.e., `huggingface-cli login`).

```
python build_broad.py \
    --imagenet-based \
    --coco-based \
    --partial_broad_cache_dir <PATH> \
    --imagenet_cache_dir <PATH> \
    --coco_cache_dir <PATH> \ 
    --output_dir <PATH> \ 
    --hub_push_path <hf_account/repo_name>
```

The bulk of the building time is given by the downloading of the two auxiliary datasets, and a couple of hours should be expected with a reasonable download speed.

[^1]: For licensing reasons, the output of `build_broad.py` cannot be uploaded to a public huggingface repository.
