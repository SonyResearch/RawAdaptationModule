import os
import os.path as osp

import rawpy
from tqdm import tqdm

import numpy as np
import cv2
import argparse


# info
# dark level = 2047
# white level = 16383

def opts():
    parser = argparse.ArgumentParser(description='CR2 RAW file to numpy RAW')
    parser.add_argument('--input_folder', default='./LOD/RAW')
    parser.add_argument('--dark_output_folder', default='./LOD/dark_raw')
    parser.add_argument('--normal_output_folder', default='./LOD/normal_raw')
    parser.add_argument('--h', type=int, default=800,
                        help="if w/o resize: 2251, half size: 1125")
    parser.add_argument('--w', type=int, default=1200,
                        help="if w/o resize: 3372, half_size: 1686")
    return parser.parse_args()


def pack_raw_bayer(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    red = im[R[0][0]:H:2, R[1][0]:W:2]
    green1 = im[G1[0][0]:H:2, G1[1][0]:W:2]
    green2 = im[G2[0][0]:H:2, G2[1][0]:W:2]
    blue = im[B[0][0]:H:2, B[1][0]:W:2]
    out = np.stack((red, green1, green2, blue), axis=0)

    img = out.astype(int)  # uint->int
    img = img - 2047
    img = img / 14336
    img = img.transpose((1, 2, 0))
    return img


if __name__ == "__main__":
    args = opts()
    input_folder = args.input_folder
    dark_output_folder = args.dark_output_folder
    normal_output_folder = args.normal_output_folder
    if not osp.exists(normal_output_folder):
        os.mkdir(normal_output_folder)
    if not osp.exists(dark_output_folder):
        os.mkdir(dark_output_folder)

    for filename in tqdm(sorted(os.listdir(input_folder))):
        if filename[-3:] == 'CR2':
            number = filename[:-4]

            if osp.exists(osp.join(input_folder, filename)):

                raw = rawpy.imread(osp.join(input_folder, filename))
                img = pack_raw_bayer(raw)

                img = cv2.resize(img, (args.w, args.h))
                if int(number) % 2 == 0:
                    np.save(osp.join(dark_output_folder, filename[:-4] + '.npy'), img)
                else:
                    np.save(osp.join(normal_output_folder, filename[:-4] + '.npy'), img)