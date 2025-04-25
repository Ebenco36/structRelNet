import pandas as pd
from datasets import load_dataset
from pathlib import Path

class ToTToDataset:
    def __init__(self, output_dir="data/processed/totto", sample_size=500):
        self.dataset_name = "totto"
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size
        self.data = None

    def load(self):
        print("Loading dataset...")
        self.data = load_dataset(self.dataset_name, split=f"train[:{self.sample_size}]")

    def process_and_save(self):
        print("Processing and saving data...")
        split_dir = self.output_dir / "train_tables"
        split_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        for i, example in enumerate(self.data):
            table_data = example["table"]
            if isinstance(table_data, list) and len(table_data) > 1:
                header = table_data[0]
                rows = table_data[1:]
                try:
                    table = pd.DataFrame(rows, columns=header)
                except Exception as e:
                    print(f"⚠️ Skipping malformed table {i}: {e}")
                    continue
            else:
                continue

            tid = f"totto_train_{i:05d}"
            table_path = split_dir / f"{tid}.csv"
            table.to_csv(table_path, index=False)

            # Safely extract sentence
            sentence = ""
            annotations = example.get("sentence_annotations")
            if isinstance(annotations, list) and len(annotations) > 0:
                first = annotations[0]
                if isinstance(first, dict):
                    sentence = first.get("final_sentence", "")

            metadata.append({
                "id": tid,
                "path": str(table_path.relative_to(self.output_dir)),
                "meta": {
                    "sentence": sentence
                }
            })

        pd.DataFrame(metadata).to_csv(self.output_dir / "train_metadata.csv", index=False)
        print(f"Saved ToTTo train split with {len(metadata)} tables.")

    def run(self):
        self.load()
        self.process_and_save()