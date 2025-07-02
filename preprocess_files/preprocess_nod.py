import argparse
from pathlib import Path
import numpy as np
import rawpy
import cv2
from tqdm import tqdm

def resize(data, target_height, target_width):
    ori_height, ori_width = data.shape[:2]

    if target_width is None:
        target_width = int(ori_width / ori_height * target_height)

    data = cv2.resize(data, (target_width, target_height))

    return data


def bayer2rggb(src_path):
    raw_file = rawpy.imread(str(src_path))
    bayer_image = raw_file.raw_image

    red = bayer_image[0::2, 0::2]
    green_red = bayer_image[0::2, 1::2]
    green_blue = bayer_image[1::2, 0::2]
    blue = bayer_image[1::2, 1::2]

    raw_rggb = np.stack((red, green_red, green_blue, blue), axis=-1)
    return raw_rggb


def process_folder(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    raw_files = list(input_folder.glob("*.NEF"))

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
