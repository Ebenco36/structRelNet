import pandas as pd
from datasets import load_dataset
from pathlib import Path

class TabFactDataset:
    def __init__(self, output_dir="data/processed/tabfact", seed=42):
        self.dataset_name = "tab_fact"
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.splits = {}

    def load(self):
        print("Loading dataset...")
        raw = load_dataset(self.dataset_name, "tab_fact")
        self.splits = {
            "train": raw["train"],
            "val": raw["validation"],
            "test": raw["test"]
        }

    def process_and_save(self):
        print("Processing and saving data...")
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / f"{split}_tables"
            split_dir.mkdir(parents=True, exist_ok=True)

            metadata = []
            for i, example in enumerate(self.splits[split]):
                tid = f"tabfact_{split}_{i:05d}"
                label = "entailed" if example["label"] == 1 else "refuted"
                metadata.append({
                    "id": tid,
                    "path": "N/A",
                    "meta": {
                        "statement": example["statement"],
                        "label": label
                    }
                })

            pd.DataFrame(metadata).to_csv(self.output_dir / f"{split}_metadata.csv", index=False)
            print(f"Saved {split} split with {len(metadata)} examples.")

    def run(self):
        self.load()
        self.process_and_save()