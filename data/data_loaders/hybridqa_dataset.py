import pandas as pd
from datasets import load_dataset
from pathlib import Path

class HybridQADataset:
    def __init__(self, output_dir="data/processed/hybridqa", seed=42):
        self.dataset_name = "hybrid_qa"
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.splits = {}

    def load(self):
        print("Loading dataset...")
        raw = load_dataset(self.dataset_name)
        self.splits = {
            "train": raw["train"],
            "val": raw["validation"],
            "test": raw["test"]
        }

    def process_and_save(self):
        print("Processing and saving data...")
        for split in ["train", "val", "test"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            metadata = []

            for i, example in enumerate(self.splits[split]):
                tid = f"hybridqa_{split}_{i:05d}"
                metadata.append({
                    "id": tid,
                    "path": "N/A",
                    "meta": {
                        "question": example["question"],
                        "answer": example["answer_text"]
                    }
                })

            pd.DataFrame(metadata).to_csv(self.output_dir / f"{split}_metadata.csv", index=False)
            print(f"Saved {split} split with {len(metadata)} examples.")

    def run(self):
        self.load()
        self.process_and_save()