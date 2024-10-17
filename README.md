
# HOI-Swap: Swapping Objects in Videos with Hand-Object Interaction Awareness

[**HOI-Swap: Swapping Objects in Videos with Hand-Object Interaction Awareness**](https://arxiv.org/abs/2406.07754)   
Zihui Xue, Mi Luo, Changan Chen, Kristen Grauman  
NeurIPS, 2024  
[project page](https://vision.cs.utexas.edu/projects/HOI-Swap/) | [arxiv](https://arxiv.org/abs/2406.07754) | [bibtex](#citation)

## News

- **10/2024** Unfortunately, we are unable to release the HOI-Swap pre-trained checkpoints due to legal constraints. However, the HOI-Swap edit benchmark and evaluation code are now available [here]((#hoi-swap-edit-benchmark)). Stay tuned for the training and inference code.

## HOI-Swap edit benchmark

The benchmark includes both image and video editing tasks. You can download the data [here](https://drive.google.com/file/d/1b0BpBXj6UVvL-PygpV-TEonsbU8nP-Pj/view?usp=share_link). The source images/videos and reference object images used as model input for editing are provided. Alongside, we provide HOI-Swap's generated results together with baseline approaches, thanks to their open-source availability! See the sections below for more details.

### Image editing

The evaluation set for image editing includes 1,250 source images, each paired with four reference object images, resulting in a total of 5,000 edited images. `images_hoi4d` contains 1000 images from HOI4D, and `images_egoexo4d` contains 250 EgoExo4D images. We provide the results from three baseline methods alongside HOI-Swap. Additionally, our evaluation requires using the [hand object detector](https://github.com/ddshan/hand_object_detector.git). To simplify the process, we've already included preprocessed detection results (found in the `hand_det` folder).

Evaluation: Run `evaluation/eval_image.py` for quantitative evaluations (Table 1 of the paper).

Baselines:

- Paint-by-example: https://github.com/Fantasy-Studio/Paint-by-Example
- AnyDoor: https://github.com/ali-vilab/AnyDoor
- Affordance Diffusion: https://github.com/NVlabs/affordance_diffusion

### Video editing
The video editing evaluation set consists of 25 source videos, each combined with four reference object images, yielding 100 unique edited videos. `videos_hoi4d` contains 17 videos from HOI4D, and `videos_ood` contains 8 videos from TCN Pouring and EPIC-Kitchens, demonstrating zero-shot generalization capabilities.

We also provide preprocessed detection results using the [hand object detector](https://github.com/ddshan/hand_object_detector.git), available in the hand_det_video folder.

Evaluation:

- Run [VBench](https://github.com/Vchitect/VBench) with `--dimension subject_consistency motion_smoothness --mode custom_input` for the first 2 metrics in Table 1
- Run `evaluation/eval_video.py` for the last 3 metrics in Table 1.

Baselines:

- AnyDoor for every frame: https://github.com/ali-vilab/AnyDoor
- AnyDoor + AnyV2V: https://github.com/TIGER-AI-Lab/AnyV2V
- VideoSwap: https://github.com/showlab/VideoSwap

## Disclaimer

This repository provides a personal reproduction of HOISwap, completed independently at the University of Texas at Austin. The codebase is released as a personal project and is not affiliated with any external organizations.

## Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@article{xue2024hoi,
  title={HOI-Swap: Swapping Objects in Videos with Hand-Object Interaction Awareness},
  author={Xue, Zihui and Luo, Mi and Chen, Changan and Grauman, Kristen},
  journal={arXiv preprint arXiv:2406.07754},
  year={2024}
}
```