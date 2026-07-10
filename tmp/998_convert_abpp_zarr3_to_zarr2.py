import argparse
import shutil
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from tqdm import tqdm


def get_compressor(name):
    if name == "none":
        return None
    if name == "zstd":
        return numcodecs.Zstd(level=3)
    if name == "blosc-zstd":
        return numcodecs.Blosc(
            cname="zstd",
            clevel=3,
            shuffle=numcodecs.Blosc.BITSHUFFLE,
        )
    raise ValueError(f"Unsupported compressor: {name}")


def copy_attrs(src, dst):
    try:
        dst.attrs.update(dict(src.attrs))
    except Exception:
        pass


def is_array(node):
    return hasattr(node, "shape") and hasattr(node, "dtype")


def copy_array(src, dst_group, name, batch_size, compressor):
    chunks = tuple(src.chunks) if src.chunks is not None else None
    dst = dst_group.create_array(
        name=name,
        shape=src.shape,
        chunks=chunks,
        dtype=src.dtype,
        compressor=compressor,
        overwrite=True,
    )
    copy_attrs(src, dst)

    if len(src.shape) == 0:
        dst[...] = np.asarray(src[...])
        return

    n = src.shape[0]
    desc = dst.path or name
    for start in tqdm(range(0, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        dst[start:end] = np.asarray(src[start:end])


def copy_group(src_group, dst_group, batch_size, compressor):
    copy_attrs(src_group, dst_group)
    for key in src_group.keys():
        src_child = src_group[key]
        if is_array(src_child):
            copy_array(src_child, dst_group, key, batch_size, compressor)
        else:
            dst_child = dst_group.create_group(key, overwrite=True)
            copy_group(src_child, dst_child, batch_size, compressor)


def main():
    parser = argparse.ArgumentParser(
        description="Convert an ABPP Zarr v3 directory to a Zarr v2 directory.")
    parser.add_argument(
        "--src",
        default="/home/ubuntu/AutoBioPlus/logs/task/ABPP-XHand-Tube-Insert-120x160.zarr",
    )
    parser.add_argument(
        "--dst",
        default="/home/ubuntu/AutoBioPlus/logs/task/ABPP-XHand-Tube-Insert-120x160-v2.zarr",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--compressor",
        choices=["blosc-zstd", "zstd", "none"],
        default="none",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if dst_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{dst_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst_path)

    src_root = zarr.open(str(src_path), mode="r", zarr_format=3)
    dst_root = zarr.open(str(dst_path), mode="w", zarr_format=2)
    copy_group(
        src_group=src_root,
        dst_group=dst_root,
        batch_size=args.batch_size,
        compressor=get_compressor(args.compressor),
    )
    print(f"Wrote Zarr v2 dataset to {dst_path}")


if __name__ == "__main__":
    main()
