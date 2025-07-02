import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def bayer2rggb(path):
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw[0::3] + raw[1::3] * 256 + raw[2::3] * 65536
    raw = raw.reshape((1856, 2880)).astype(np.float32)
    raw_norm = raw / (2 ** 24 - 1)

    red = raw_norm[0::2, 0::2]
    green_red = raw_norm[0::2, 1::2]
    green_blue = raw_norm[1::2, 0::2]
    blue = raw_norm[1::2, 1::2]

    raw_rggb = np.stack((red, green_red, green_blue, blue), axis=-1)
    return raw_rggb


def process_folder(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_files = list(input_folder.glob("*.raw"))

    for raw_file in tqdm(raw_files):
        rggb_raw = bayer2rggb(raw_file)
        output_filename = raw_file.with_suffix(".npy").name
        np.save(output_folder / output_filename, rggb_raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RAW Bayer files to RGGB .npy format.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder with .raw files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the output .npy files")

    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder)
