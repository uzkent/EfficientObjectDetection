import numpy as np
import json
import glob
import pdb

def find_difference(gt_patch, preds_patch):
    diff_all = []
    try:
        for ind in range(len(preds_patch)):
            preds_ind = preds_patch[ind]
            preds_ind[:4] = preds_ind[:4] / 416.
            diff_all.append(np.sum(np.abs(gt_patch[0] - preds_ind[:4])))
    except Exception:
        preds_patch[:4] = preds_patch[:4] / 416.
        diff_all.append(np.sum(np.abs(gt_patch[0] - preds_patch[:4])))

    if not diff_all:
        diff = 3
    else:
        diff = np.min(diff_all)
    return diff

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)
    labels = []
    for dt in data:
        labels.append(dt[-1])

    return data, labels

def read_groundtruth(image_paths, base_dir_gt, base_dir_pred_fd, base_dir_pred_cd):
    object_interest = [73, 74, 76, 79, 77, 71, 72]
    num_windows = 2
    for index, img_path in enumerate(image_paths):
        counter = 0
        diff_fd = np.zeros((num_windows, num_windows))
        diff_cd = np.zeros((num_windows, num_windows))
        img_id = img_path.split('/')[-1].split('.')[0]
        for xind in range(num_windows):
            for yind in range(num_windows):
                try:
                    gt_bboxes, gt_labels = read_json('{}{}_{}_{}{}'.format(base_dir_gt, img_id, str(xind), str(yind),'.json'))
                except Exception as error:
                    print(error)
                    continue
                if not gt_bboxes:
                    diff_cd[xind, yind] = 0
                    diff_fd[xind, yind] = 3
                    continue
                try:
                    preds_fd = np.loadtxt('{}{}_{}_{}'.format(base_dir_pred_fd, img_id, str(xind), str(yind)))
                except Exception:
                    preds_fd = []
                try:
                    preds_cd = np.loadtxt('{}{}_{}_{}_ds'.format(base_dir_pred_cd, img_id, str(xind), str(yind)))
                except Exception:
                    preds_cd = []
                for index, lb in enumerate(gt_labels):
                    if lb in object_interest:
                        diff_fd[xind, yind] += find_difference(gt_bboxes[index], preds_fd)
                        diff_cd[xind, yind] += find_difference(gt_bboxes[index], preds_cd)

        np.savetxt('{}{}'.format('/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_fd_small/', img_id), diff_fd)
        np.savetxt('{}{}'.format('/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_cd_small/', img_id), diff_cd)

    return diff_fd, diff_cd

base_dir_gt = '/atlas/u/buzkent/Single_Shot_Object_Detector/prepare_dataset/small_chips_detector/'
base_dir_pred_fd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/fd_output_txt_small/'
base_dir_pred_cd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/cd_output_txt_small/'
image_paths = glob.glob('/atlas/u/buzkent/Single_Shot_Object_Detector/prepare_dataset/train_chips_xview/*.jpg')

read_groundtruth(image_paths, base_dir_gt, base_dir_pred_fd, base_dir_pred_cd)
