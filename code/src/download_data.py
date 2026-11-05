"""Download a prespecified subset of Zhang's CC-BY benchmark, verifying hashes."""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'

def main():
    manifest = json.loads((RAW / 'file_manifest.json').read_text())
    selected = []
    for row in manifest:
        parts = row['filename'].removesuffix('.json').split('-')
        if len(parts) == 4 and int(parts[3]) < 12:
            selected.append(row)
        elif row['filename'] in ['retrieval_probability.json', 'example.json']:
            selected.append(row)
    (RAW / 'instances').mkdir(exist_ok=True)
    def get(row):
        name = row['filename']
        dest = RAW / ('instances/' + name if name.count('-') == 3 else name)
        expected = row['content_details']['sha256_hash']
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
            return name
        for attempt in range(4):
            try:
                response = requests.get(row['content_details']['download_url'], timeout=35)
                response.raise_for_status()
                assert hashlib.sha256(response.content).hexdigest() == expected
                json.loads(response.content)
                dest.write_bytes(response.content)
                return name
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, name in enumerate(pool.map(get, selected)):
            if i % 80 == 0:
                print(f'{i + 1}/{len(selected)} downloaded', flush=True)
    audit = {
        'source': 'https://data.mendeley.com/datasets/3j52cjrm52/3',
        'doi': '10.17632/3j52cjrm52.3', 'author': 'Zhanluo Zhang',
        'license': 'CC BY 4.0',
        'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'selection': 'All 56 shape/density groups, instance suffix 0..11, selected before outcome inspection.',
        'split': {'train': 'suffix 0..5', 'validation': 'suffix 6..7', 'test': 'suffix 8..11'},
        'files': [{'name': f['filename'], 'sha256': f['content_details']['sha256_hash'],
                   'bytes': f['size']} for f in selected],
        'changes': 'Original files unchanged. Normalized probability rows and independent analysis are derived works.',
        'note': 'Research benchmark scenarios, not Incheon operational container records.'
    }
    (RAW / 'provenance.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print('Downloaded and hash-verified', len(selected), flush=True)

if __name__ == '__main__':
    main()
