import pandas as pd
from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

class WikiTableQuestionsDataset:
    def __init__(self, output_dir="data/processed/wikitablequestions", seed=42):
        self.dataset_name = "wikitablequestions"
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.splits = {}

    def load(self):
        print("Loading dataset...")
        raw = load_dataset(self.dataset_name)
        all_data = raw["train"].train_test_split(test_size=0.2, seed=self.seed)
        train_val = all_data["train"].train_test_split(test_size=0.1, seed=self.seed)

        self.splits = {
            "train": train_val["train"],
            "val": train_val["test"],
            "test": all_data["test"]
        }

    def process_and_save(self):
        print("Processing and saving data...")
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / f"{split}_tables"
            split_dir.mkdir(parents=True, exist_ok=True)

            metadata = []
            for i, example in enumerate(self.splits[split]):
                table = pd.DataFrame(example["table"]["rows"], columns=example["table"]["header"])
                tid = f"wtq_{split}_{i:05d}"
                table_path = split_dir / f"{tid}.csv"
                table.to_csv(table_path, index=False)

                metadata.append({
                    "id": tid,
                    "path": str(table_path.relative_to(self.output_dir)),
                    "meta": {
                        "question": example["question"],
                        "answers": example["answers"]
                    }
                })

            pd.DataFrame(metadata).to_csv(self.output_dir / f"{split}_metadata.csv", index=False)
            print(f"Saved {split} split with {len(metadata)} tables.")

    def run(self):
        self.load()
        self.process_and_save()