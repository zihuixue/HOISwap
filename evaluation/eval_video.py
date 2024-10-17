import glob
import numpy as np
from evaluation.eval_image import calculate_iou


def load_det_list(det_dir):
    det_file = sorted(glob.glob(f"{det_dir}/*.npy"))
    det = [np.load(f) for f in det_file]
    assert len(det) == 14, f"{det_dir} has {len(det)} frames"
    return det


def eval_two_lists(det_list1, det_list2):
    miou_score = []
    contact_score = []
    hand_score = []
    for det1, det2 in zip(det_list1, det_list2):
        miou = calculate_iou(det1[:4], det2[:4])
        miou_score.append(miou)
        contact_score.append(det1[5] == det2[5])
        hand_score.append(det2[4])
    return np.mean(contact_score), np.mean(miou_score), np.mean(hand_score)


def eval():
    video_list = glob.glob(f'{root_dir}/videos_hoi4d/**/source.mp4', recursive=True) + \
                    glob.glob(f'{root_dir}/videos_ood/**/source.mp4', recursive=True)
    print(f'Found {len(video_list)} videos')
    method_dir = ['baseline1', 'baseline2', 'baseline3', 'ours'] # baseline1: per-frame, baseline2: anyv2v, baseline3: videoswap
    result = {m: [] for m in method_dir}
    for video in video_list:
        det_path = video.replace('.mp4', '').replace('/videos_', 'hand_det_video/videos_')
        gt_det = load_det_list(det_path)
        for m in method_dir:
            det = load_det_list(det_path.replace('source', m))
            contact_score, miou_score, hand_score = eval_two_lists(gt_det, det)
            result[m].append([contact_score, miou_score, hand_score])
    print('method, contact agreement, hand agreement, hand fidelity')
    for m in method_dir:
        result[m] = np.mean(result[m], axis=0)
        print(f"{m} {result[m]}")


if __name__ == '__main__':
    root_dir = 'edit_benchmark'  # replace with your data path
    eval()