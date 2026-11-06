"""Check the downloaded files against the SHA-256 digests this study was run on."""
import json
import os
import sys

from common import DATA_DIR, files, sha256

expected = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "data_sha256.json")))
bad = 0
for path in files():
    rel = os.path.relpath(path, DATA_DIR).replace(os.sep, "/")
    if expected.get(rel) != sha256(path):
        print("MISMATCH", rel)
        bad += 1
print(f"{len(expected) - bad} of {len(expected)} files match the digests used in this study.")
sys.exit(1 if bad else 0)
