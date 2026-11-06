"""Download the three public AuditBench Petri datasets used in this study into ./data."""
from huggingface_hub import snapshot_download

DATASETS = {
    "transcripts_all": "auditing-agents/petri-transcripts-all-llama70b",
    "summaries_all": "auditing-agents/petri-judge-summaries-all-llama70b",
    "top50": "auditing-agents/petri-transcripts-top50-llama70b",
}

if __name__ == "__main__":
    for folder, repo in DATASETS.items():
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=f"data/{folder}")
    print("Done. Now run: python src/verify_data.py && python src/analysis.py && python src/figures.py")
