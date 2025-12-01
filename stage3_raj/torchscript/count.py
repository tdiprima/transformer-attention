from raj_dataset import RajDataset

ds = RajDataset(root_dir="/data/erich/raj/data/train")
print(f"Total images: {len(ds)}")  # 72102
print(f"Classes: {ds.num_classes()}")  # 10

# find /data/erich/raj/data/train -name "*.png" | wc -l
