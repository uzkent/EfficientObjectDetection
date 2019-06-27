import numpy as np
import json
import glob
import pdb

def xyxy2xywh(x):
    y = np.zeros((x.shape))
    x = x / 416.
    y[0] = (x[0] + x[2]) / 2.
    y[1] = (x[1] + x[3]) / 2.
    y[2] = x[2] - x[0]
    y[3] = x[3] - x[1]
    return y

def find_difference(gt_patch, preds_patch):
    diff_all = []
    try:
        for ind in range(len(preds_patch)):
            preds_ind = preds_patch[ind]
            preds_ind[:4] = xyxy2xywh(preds_ind[:4])
            diff_all.append(np.sum(np.abs(gt_patch[0][1:] - preds_ind[:4])))
    except Exception:
        preds_patch[:4] = xyxy2xywh(preds_patch[:4])
        diff_all.append(np.sum(np.abs(gt_patch[0][1:] - preds_patch[:4])))

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
    for index, img_path in enumerate(image_paths):
        counter = 0
        diff_fd = np.zeros((num_windows, num_windows))
        diff_cd = np.zeros((num_windows, num_windows))
        img_id = img_path.split('/')[-1].split('.')[0]
        for xind in range(num_windows):
            for yind in range(num_windows):
                """
                try:
                    gt_bboxes, gt_labels = read_json('{}{}_{}_{}{}'.format(base_dir_gt, img_id, str(xind), str(yind),'.json'))
                except Exception as error:
                    diff_cd[xind, yind] = 0
                    diff_fd[xind, yind] = 3
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
                    diff_fd[xind, yind] += find_difference(gt_bboxes[index], preds_fd)
                    diff_cd[xind, yind] += find_difference(gt_bboxes[index], preds_cd)
                """
                try:
                    gt_bboxes = np.loadtxt('{}{}_{}_{}.txt'.format(base_dir_gt, img_id, str(xind), str(yind))).reshape([-1,5])
                except Exception as error:
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
                for index in enumerate(range(gt_bboxes.shape[0])):
                    diff_fd[xind, yind] += find_difference(gt_bboxes[index, :], preds_fd)
                    diff_cd[xind, yind] += find_difference(gt_bboxes[index, :], preds_cd)
        np.savetxt('{}{}'.format('/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_fd/', img_id), diff_fd)
        np.savetxt('{}{}'.format('/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_cd/', img_id), diff_cd)

    return diff_fd, diff_cd

num_windows = 4
base_dir_gt = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/labels/'
base_dir_pred_fd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/fd_output_txt/'
base_dir_pred_cd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/cd_output_txt/'
image_paths = glob.glob('/atlas/u/buzkent/Single_Shot_Object_Detector/prepare_dataset/train_images/*.tif')

read_groundtruth(image_paths, base_dir_gt, base_dir_pred_fd, base_dir_pred_cd)
