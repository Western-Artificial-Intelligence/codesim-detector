import json
from itertools import islice
from datasets import load_dataset


py_ds = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",        # <-- only Python files
    split="train",
    streaming=True
)


for i, s in enumerate(py_ds):
    print(s.get("path") or s.get("max_stars_repo_path") or "<no-path>")
    print(s.get("content")[:2000])
    print("-"*50)
    if i >= 4:
        break


subset = list(islice(py_ds, 5000))  # first 5k Python files


with open("python_subset.jsonl", "w", encoding="utf-8") as f:
    for ex in subset:
        rec = {
            "path": ex.get("path") or ex.get("max_stars_repo_path"),
            "code": ex.get("content")
        }
        json.dump(rec, f, ensure_ascii=False)
        f.write("\n")
