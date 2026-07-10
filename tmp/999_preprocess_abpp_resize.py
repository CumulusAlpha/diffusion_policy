import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import zarr
from tqdm import tqdm


def get_array_compression_kwargs(src):
    if hasattr(src, "compressors"):
        compressors = src.compressors
        if compressors:
            return {"compressors": compressors}
        return {}
    return {"compressor": src.compressor}


def copy_array(src_group, dst_group, key):
    src = src_group[key]
    dst_group.create_array(
        name=key,
        data=src[:],
        chunks=src.chunks,
        **get_array_compression_kwargs(src),
        overwrite=True,
    )


def resize_rgb(src_rgb, dst_rgb, height, width, batch_size):
    n_steps = src_rgb.shape[0]
    n_cameras = src_rgb.shape[1]

    for start in tqdm(range(0, n_steps, batch_size), desc="resize rgb"):
        end = min(start + batch_size, n_steps)
        batch = src_rgb[start:end]
        out = np.empty(
            (end - start, n_cameras, height, width, 3),
            dtype=np.uint8,
        )
        for t in range(end - start):
            for camera_idx in range(n_cameras):
                out[t, camera_idx] = cv2.resize(
                    batch[t, camera_idx],
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
        dst_rgb[start:end] = out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/home/ubuntu/AutoBioPlus/logs/task/ABPP-XHand-Tube-Insert.zarr",
    )
    parser.add_argument(
        "--dst",
        default="/home/ubuntu/AutoBioPlus/logs/task/ABPP-XHand-Tube-Insert-120x160.zarr",
    )
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    if dst_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{dst_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst_path)

    src_root = zarr.open(str(src_path), mode="r")
    dst_root = zarr.open(str(dst_path), mode="w")
    src_data = src_root["data"]
    src_meta = src_root["meta"]
    dst_data = dst_root.create_group("data")
    dst_meta = dst_root.create_group("meta")

    for key in ["action", "state", "timestamp"]:
        copy_array(src_data, dst_data, key)

    for key in src_meta.keys():
        copy_array(src_meta, dst_meta, key)

    src_rgb = src_data["rgb"]
    dst_data.create_array(
        name="rgb",
        shape=(src_rgb.shape[0], src_rgb.shape[1], args.height, args.width, 3),
        chunks=(src_rgb.chunks[0], src_rgb.shape[1], args.height, args.width, 3),
        dtype=src_rgb.dtype,
        **get_array_compression_kwargs(src_rgb),
        overwrite=True,
    )
    resize_rgb(
        src_rgb=src_rgb,
        dst_rgb=dst_data["rgb"],
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
    )

    print(f"Wrote resized dataset to {dst_path}")


if __name__ == "__main__":
    main()
