import re
import os
import torch
from PIL import Image
import json
import numpy as np

from scripts.metrics import ImageMetricsCalculator, calculate_text_metrics

def tensor_to_pil(tensor):
    tensor = tensor.cpu().float()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val > min_val:
        tensor = (tensor - min_val) / (max_val - min_val)
    tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
    np_array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(np_array)

class VisualizationEvaluator():
    def __init__(self, **kwargs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Evaluator initialized on device: {self.device}")

        self.image_metric_calculator = ImageMetricsCalculator(device=self.device)

    def evaluate(self, text_preds, gt_data, sketch_preds, section, finish):
        action_results = []
        viz_results = []
        all_text_preds = []
        all_text_gts = []

        train_tasks = [i.get('train_task', None) for i in gt_data]
        scene_ids = [i.get('scene_id') for i in gt_data]
        ids = [i.get('idx') for i in gt_data]

        for idx, task in enumerate(train_tasks):
            if task == "instruction_gen_visualcues":
                pred_instruction = text_preds[idx]
                goal_instruction = gt_data[idx].get('label_text')

                all_text_preds.append(pred_instruction)
                all_text_gts.append([goal_instruction])
                
                action_results.append({
                    "idx": ids[idx],
                    "scene_id": scene_ids[idx],
                    "pred": pred_instruction,
                    "gt": goal_instruction
                })
            elif task == "single_step_visualization":
                pred_img_tensor = sketch_preds[idx]
                goal_img_pil = gt_data[idx].get('label_imgs', [None])[-1]
                pred_img_pil = tensor_to_pil(pred_img_tensor)

                image_scores = self.image_metric_calculator.calculate(pred_img_pil, goal_img_pil)
                print(f"[Visual Task] Scores for {scene_ids[idx]}: {image_scores}")
        
                viz_results.append({
                        "idx": ids[idx],
                        "scene_id": scene_ids[idx],
                        "scores": image_scores
                    })

        metrics = {}

        # ins
        print(f"Calculating metrics for {len(all_text_preds)} text predictions...")
        text_metrics = calculate_text_metrics(all_text_preds, all_text_gts)
        metrics.update(text_metrics)

        # vis
        print(f"Calculating average metrics for {len(viz_results)} image predictions...")
        score_lists = {key: [] for key in viz_results[0]['scores'].keys()}
        
        for res in viz_results:
            for key, value in res['scores'].items():
                score_lists[key].append(value)
        
        for key, values in score_lists.items():
            if values:
                metrics[f"eval_vis_{key}"] = np.mean(values)

        print(f"Metrics calculated: {metrics}")
        return metrics
