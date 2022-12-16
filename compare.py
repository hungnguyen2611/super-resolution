import glob
import os
import time
from collections import OrderedDict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from natsort import natsort
from tqdm import tqdm


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))




if __name__ == "__main__":
    out_data_path = fiFindByWildcard("./results_crop (1)/out/*")
    gt_data_path = fiFindByWildcard("./results_crop (1)/target/*")
    source_data_path = fiFindByWildcard("./results_crop (1)/source/*")

    for src_path, out_path, gt_path in tqdm(list(zip(source_data_path, out_data_path, gt_data_path))):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title("Bicubic")
        ax2.set_title("Baseline")
        ax3.set_title("Ground truth")
        
        src = cv2.imread(src_path)[:, :, [2, 1, 0]]
        out = cv2.imread(out_path)[:, :, [2, 1, 0]]
        gt = cv2.imread(gt_path)[:, :, [2, 1, 0]]
        src = cv2.resize(src, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])
        ax1.imshow(src)
        ax2.imshow(out)
        ax3.imshow(gt)
        fig.savefig(f"./result_compare_crop_new/{os.path.basename(gt_path)}", bbox_inches='tight' , dpi=1200)
        plt.close()




