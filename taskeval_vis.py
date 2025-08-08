import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

# ==============================================================================
#  Configuration
# ==============================================================================
# Model output directory (contains datasets like vln_val_seen, etc.)
BASE_DIR = './output/20000socialimage_seq_len-784-anole-hyper-train1val1lr0.0002-R2R_Goal-prompt_anole-42/predict_task_level_results/data_samples'

# Ground Truth data directory (parallel structure to BASE_DIR)
GT_BASE_DIR = './data_samples'

# Compute device
DEVICE = 'cuda:1'
# ==============================================================================

try:
    from scripts.metrics import ImageMetricsCalculator, calculate_text_metrics
    from scripts.bleu import compute_bleu
except ImportError:
    print("Error: Could not import necessary metric modules.")
    print("Please ensure 'scripts/metrics.py' and 'scripts/bleu.py' are accessible.")
    exit(1)

# --- Helper Functions ---

def compute_sentence_bleu_4_avg(references, predictions):
    scores = []
    for ref_list, pred in zip(references, predictions):
        if not pred:
            scores.append(0.0)
            continue
        tokenized_pred = pred.split()
        tokenized_refs = [r.split() for r in ref_list]
        bleu_score, *_ = compute_bleu([tokenized_refs], [tokenized_pred], max_order=4, smooth=True)
        scores.append(bleu_score)
    return sum(scores) / len(scores) if scores else 0

def find_latest_step_file(directory, pattern_template):
    max_step = -1
    latest_file_path = None
    pattern = re.compile(pattern_template.format(r'(\d+)'))
    if not os.path.isdir(directory): return None
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            step_num = int(match.group(1))
            if step_num > max_step:
                max_step = step_num
                latest_file_path = os.path.join(directory, filename)
    return latest_file_path

def calculate_average_metrics(results_list):
    if not results_list or not isinstance(results_list[0], dict): return {}
    avg_metrics = {}
    metric_keys = results_list[0].keys()
    for key in metric_keys:
        values = [res[key] for res in results_list]
        avg_metrics[key] = np.mean(values)
    return avg_metrics

def read_pred_instruction(file_path):
    if not file_path or not os.path.exists(file_path): return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readline().strip()

def read_gt_from_json(file_path):
    """Reads the 'instruction' key from a given JSON file."""
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("instruction", "").strip()
    except (json.JSONDecodeError, KeyError):
        return None

# --- Main Execution ---
def main():
    print(f"Starting evaluation...")
    print(f"Model Output Dir: {BASE_DIR}")
    print(f"Ground Truth Dir: {GT_BASE_DIR}")
    
    image_calculator = ImageMetricsCalculator(device=DEVICE)

    dataset_dirs = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
    if not dataset_dirs:
        print(f"Error: No dataset folders found in {BASE_DIR}.")
        return
        
    print(f"Found datasets: {', '.join(dataset_dirs)}")

    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(BASE_DIR, dataset_name)
        print("\n" + "="*80)
        print(f"Processing Dataset: {dataset_name}")
        print("="*80)
        
        all_image_results = {'interleaved': [], 'one_pass': []}
        all_text_predictions = {'interleaved': [], 'one_pass': []}
        all_text_references = {'interleaved': [], 'one_pass': []}
        
        episode_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        if not episode_dirs:
            print("  -> No episode folders found in this dataset, skipping.")
            continue

        for episode in tqdm(episode_dirs, desc=f"  -> Episodes in {dataset_name}"):
            # --- Path construction for both model output and GT ---
            model_episode_path = os.path.join(dataset_path, episode)
            goal_img_path = os.path.join(model_episode_path, 'goal_obs.png')
            
            # Construct the path to the ground truth ins.json file
            gt_ins_path = os.path.join(GT_BASE_DIR, dataset_name, episode, 'ins.json')
            
            if not os.path.exists(goal_img_path): continue
            
            goal_image = Image.open(goal_img_path).convert('RGB')
            # Read GT from the new JSON source
            gt_instruction = read_gt_from_json(gt_ins_path)
            
            for method in ['interleaved', 'one_pass']:
                method_dir = os.path.join(model_episode_path, method)
                if not os.path.isdir(method_dir): continue

                pred_img_path = find_latest_step_file(method_dir, 'step_{}_obs.png')
                if pred_img_path:
                    pred_image = Image.open(pred_img_path).convert('RGB')
                    image_metrics = image_calculator.calculate(pred_image, goal_image)
                    all_image_results[method].append(image_metrics)

                pred_ins_path = None
                if method == 'interleaved':
                    pred_ins_path = find_latest_step_file(method_dir, 'step_{}_ins.txt')
                elif method == 'one_pass':
                    pred_ins_path = os.path.join(method_dir, 'final_ins.txt')
                
                # Read prediction from plain text file
                pred_instruction = read_pred_instruction(pred_ins_path)
                
                if gt_instruction and pred_instruction:
                    all_text_predictions[method].append(pred_instruction)
                    all_text_references[method].append([gt_instruction])

        print(f"\n--- Evaluation Results for {dataset_name} ---")
        for method in ['interleaved', 'one_pass']:
            num_img_eps = len(all_image_results[method])
            num_txt_eps = len(all_text_predictions[method])

            print(f"\n----- Method: '{method}' -----")
            
            if num_img_eps > 0:
                print(f"  Image Metrics (over {num_img_eps} episodes):")
                avg_img_metrics = calculate_average_metrics(all_image_results[method])
                print(f"    PSNR    : {avg_img_metrics.get('psnr', 0):.4f}")
                print(f"    SSIM    : {avg_img_metrics.get('ssim', 0):.4f}")
                print(f"    LPIPS   : {avg_img_metrics.get('lpips', 0):.4f}")
                print(f"    DreamSim: {avg_img_metrics.get('dreamsim', 0):.4f}")
            else:
                print("  No valid image results found.")
                
            if num_txt_eps > 0:
                print(f"  Instruction Metrics (over {num_txt_eps} episodes):")
                text_metrics = calculate_text_metrics(all_text_predictions[method], all_text_references[method])
                bleu_4_score = compute_sentence_bleu_4_avg(all_text_references[method], all_text_predictions[method])
                text_metrics['eval_ins_bleu'] = bleu_4_score

                print(f"    BLEU-4  : {text_metrics.get('eval_ins_bleu', 0):.4f}")
                print(f"    ROUGE-L : {text_metrics.get('eval_ins_rougeL', 0):.4f}")
                print(f"    METEOR  : {text_metrics.get('eval_ins_meteor', 0):.4f}")
                print(f"    CIDEr   : {text_metrics.get('eval_ins_cider', 0):.4f}")
                print(f"    SPICE   : {text_metrics.get('eval_ins_spice', 0):.4f}")
            else:
                print("  No valid instruction text results found.")
    
    print("\n" + "="*80)
    print("All datasets evaluated.")

if __name__ == '__main__':
    main()