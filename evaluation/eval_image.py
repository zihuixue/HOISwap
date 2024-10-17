import os
import glob
import pandas as pd
import numpy as np


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
    bbox1: tuple, (x1, y1, x2, y2) representing the coordinates of the top-left and bottom-right corners of the first bounding box.
    bbox2: tuple, (x1, y1, x2, y2) representing the coordinates of the top-left and bottom-right corners of the second bounding box.

    Returns:
    iou: float, Intersection over Union (IoU) between the two bounding boxes.
    """

    # Calculate coordinates of intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # If the boxes don't intersect, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of each bounding box
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def eval(setting):
    data_dir = os.path.join(root_dir, setting)
    df = pd.read_csv(f'{data_dir}/source/stat.csv')
    gt_dict = {}
    hand_score = []
    hand_label = []
    for i, row in df.iterrows():
        idx = row['idx']
        np_file = f"{data_dir}/hand_det/source/{row['cur_frame_path'].replace('.png', '.npy')}"
        arr = np.load(np_file)
        if arr[4] < 0.9:  # frames with high hand confidence --> frames with hand
            continue
        gt_dict[idx] = arr[:6]
        hand_score.append(gt_dict[idx][-2])
        hand_label.append(gt_dict[idx][-1])
    print(f'Evaluate source on {len(df)} samples, has hand ratio {len(gt_dict) / len(df)}\n'
        f'hand score {np.mean(hand_score)}\n'
        f'contact ratio {(np.array(hand_label) >= 3).sum() / len(hand_label)}, '
        f'label dist {np.unique(hand_label, return_counts=True)}')

    # affordance diffusion baseline file name is different from others, creating a mapping here
    affordance_files = sorted(glob.glob(f'{data_dir}/hand_det/affordance/*.npy'))
    refobj_list = df['refobj_path'].unique().tolist()
    affordance_mapping = dict(zip(refobj_list, affordance_files))
    
    method_list = ['paint_by_example', 'anydoor', 'affordance', 'ours']
    final_result = {}
    for mid, m in enumerate(method_list):
        result = {k: [] for k in ['contact_agree', 'hand_agree', 'hand_fid']}
        for i, row in df.iterrows():
            idx = row['idx']
            if idx not in gt_dict:
                continue
            if m != 'affordance':   
                np_file = f"{data_dir}/hand_det/{m}/generated_{idx}.npy"
            else:
                np_file = affordance_mapping[row['refobj_path']]
            arr = np.load(np_file)
            gt_arr = gt_dict[idx]
            det = arr[:4] if m != 'affordance' else arr[:4] * 2  # affordance diffusion baseline has a diff. resolution
            iou = calculate_iou(det, gt_arr[:4])
            result['contact_agree'].append(arr[5] == gt_arr[5])
            result['hand_agree'].append(iou)
            result['hand_fid'].append(arr[4])

        print(f'Evaluate {m} on {len(df)} samples,\n'
            f'hand fidelity {np.mean(result["hand_fid"])}\n'
            f'hand agreement {np.mean(result["hand_agree"])}\n'
            f'contact agreement {np.mean(result["contact_agree"])}')
        print('-' * 50)
        final_result[mid] = result
    return final_result
    
    
    
if __name__ == '__main__':
    root_dir = 'edit_benchmark'  # replace with your data path
    result_hoi4d = eval('images_hoi4d')
    result_egoexo4d = eval('images_egoexo4d')
    print('-' * 50)
    print('Final Result')
    # method 0-3: 'paint_by_example', 'anydoor', 'affordance', 'hoiswap'
    for mid in range(4):
        result1 = result_hoi4d[mid]
        result2 = result_egoexo4d[mid]
        for key in ['contact_agree', 'hand_agree', 'hand_fid']:
            tmp = np.mean(result1[key] + result2[key])
            print(f'method {mid}, {key}: {tmp*100:.2f}%')
        print('-' * 50)
    print('finish')