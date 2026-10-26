"""Download the public E009 checkpoint and verify its published identity."""
from pathlib import Path
import hashlib
import argparse
from common import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'checkpoints')
    args=p.parse_args()
    from huggingface_hub import hf_hub_download
    path=Path(hf_hub_download('unpackableorange/SuperMimic_Dyn','ckpt_0040000.pt',local_dir=args.directory))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    expected='182b7a5d05b6c3e4364d6223d1cf52ba18e6573563e961f6e019d6d445340c3c'
    if digest != expected: raise SystemExit(f'Checkpoint hash mismatch: {digest}; do not load it.')
    print(path, digest)
if __name__=='__main__':main()
