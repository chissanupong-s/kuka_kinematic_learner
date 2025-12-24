#!/usr/bin/env python3
import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Minimal check: load a .pt file and print basic info."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a single .pt file.",
    )
    args = parser.parse_args()
    path = args.path

    print(f"[INFO] Path argument = {path}", flush=True)

    if not os.path.exists(path):
        print(f"[ERROR] Path does not exist: {path}", flush=True)
        return

    print("[INFO] Loading with torch.load...", flush=True)
    obj = torch.load(path, map_location="cpu")
    print(f"[INFO] type(obj) = {type(obj)}", flush=True)

    # If it's a dict, show keys
    if isinstance(obj, dict):
        print(f"[INFO] dict keys = {list(obj.keys())}", flush=True)

    # If it's a tensor, show shape/dtype
    if isinstance(obj, torch.Tensor):
        print(f"[INFO] tensor shape = {obj.shape}, dtype = {obj.dtype}", flush=True)

    print("[INFO] Done.", flush=True)


if __name__ == "__main__":
    main()
