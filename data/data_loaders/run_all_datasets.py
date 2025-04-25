import os
from pathlib import Path
import sys
sys.path.append(os.getcwd())
# Import each dataset processor
from data.data_loaders.wikitablequestions_dataset import WikiTableQuestionsDataset
from data.data_loaders.tabfact_dataset import TabFactDataset
from data.data_loaders.hybridqa_dataset import HybridQADataset
from data.data_loaders.totto_dataset import ToTToDataset

def main():
    print("\n📦 Running WikiTableQuestions...")
    wtq = WikiTableQuestionsDataset(output_dir="data/processed/wikitablequestions")
    wtq.run()

    print("\n📦 Running TabFact...")
    tabfact = TabFactDataset(output_dir="data/processed/tabfact")
    tabfact.run()

    print("\n📦 Running HybridQA...")
    hybrid = HybridQADataset(output_dir="data/processed/hybridqa")
    hybrid.run()

    print("\n📦 Running ToTTo...")
    totto = ToTToDataset(output_dir="data/processed/totto", sample_size=500)
    totto.run()

    print("\n✅ All datasets processed and saved.")

if __name__ == "__main__":
    main()