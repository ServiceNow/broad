import datasets


def subcoco(coco, subcoco_id2label):
    found = set()
    image_list = []
    label_list = []
    original_filename_list = []
    original_hash_list = []
    
    for sample_index, cocoid in enumerate(coco["cocoid"]):
        if cocoid in subcoco_id2label and cocoid not in found:
            sample = coco[sample_index]
            assert sample["cocoid"] == cocoid
            found.add(cocoid)
            image_list.append(sample["image"])
            label_list.append(subcoco_id2label[cocoid])
            original_filename_list.append(sample["filename"])
            original_hash_list.append("none")
    assert len(found) == len(subcoco_id2label)
    return datasets.Dataset.from_dict({
        "image": image_list,
        "label": label_list, 
        "original_filename": original_filename_list, 
        "original_hash": original_hash_list})
