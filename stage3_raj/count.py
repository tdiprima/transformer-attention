from raj_dataset import RajDataset

ds = RajDataset(root_dir="/data/erich/raj/data/train")
print(f"Total images: {len(ds)}")
print(f"Classes: {ds.num_classes()}")

# find /data/erich/raj/data/train -name "*.png" | wc -l
