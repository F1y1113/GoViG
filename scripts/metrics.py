import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# vis
from pytorch_msssim import ssim
import lpips
from dreamsim import dreamsim

# ins
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from .bleu import compute_bleu


class ImageMetricsCalculator:
    def __init__(self, device='cuda'):
        print(f"Initializing ImageMetricsCalculator on device: {device}")
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.lpips_metric = lpips.LPIPS(net='alex').to(self.device).eval()
        self.dreamsim_metric, _ = dreamsim(pretrained=True, device=self.device)
        self.dreamsim_metric.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True)
        ])

    def _preprocess(self, image):
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        
        while image.dim() > 3:
            image = image.squeeze(0)
            
        image = image.to(self.device).float()
        
        if image.max() > 1.0:
            image = image / 255.0
            
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def calculate(self, pred_image, gt_image):
        pred_tensor_0_1 = self._preprocess(pred_image)
        goal_tensor_0_1 = self._preprocess(gt_image)

        ssim_score = ssim(pred_tensor_0_1, goal_tensor_0_1, data_range=1.0, size_average=True).item()
        mse = F.mse_loss(pred_tensor_0_1, goal_tensor_0_1)
        psnr_score = 10 * torch.log10(1.0 / mse).item() if mse > 0 else float('inf')

        pred_tensor_neg1_1 = (pred_tensor_0_1 * 2) - 1
        goal_tensor_neg1_1 = (goal_tensor_0_1 * 2) - 1

        lpips_score = self.lpips_metric(pred_tensor_neg1_1, goal_tensor_neg1_1).item()
        dreamsim_score = self.dreamsim_metric(pred_tensor_neg1_1, goal_tensor_neg1_1).item()

        return {
            'psnr': psnr_score,
            'ssim': ssim_score,
            'lpips': lpips_score,
            'dreamsim': dreamsim_score
        }


def calculate_text_metrics(predictions, references):
    """
    BLEU, ROUGE, METEOR, CIDEr, SPICE
    
    Args:
        predictions (list[str])
        references (list[list[str]]): [['ref1'], ['ref2_a', 'ref2_b'], ...]
    """
    print("Calculating text metrics...")
    metrics = {}

    # for sacrebleu
    refs_for_bleu = references
    # print(f"gt: {references}, pred:{predictions}")
    single_refs = [ref[0] for ref in references]
    # for CIDEr/SPICE
    gts = {str(i): ref for i, ref in enumerate(references)}
    res = {str(i): [pred] for i, pred in enumerate(predictions)}

    # --- BLEU ---
    bleu_scores = []
    for pred, ref_list in zip(predictions, references):
        if not pred:
            bleu_scores.append(0.0)
            continue
        tokenized_pred = pred.split()
        tokenized_refs = [r.split() for r in ref_list]
        score, *_ = compute_bleu([tokenized_refs], [tokenized_pred], max_order=4, smooth=True)
        bleu_scores.append(score)
    metrics["eval_ins_bleu"] = np.mean(bleu_scores) if bleu_scores else 0
    # sum = 0
    # for i, ref in enumerate(refs_for_bleu):
    #     print("gt:", ref, "pred:", predictions[i])
    #     sum += sacrebleu.sentence_bleu(predictions[i], ref).score
    # metrics["eval_ins_bleu"] = sum / len(predictions) if predictions else 0
    # bleu_results = sacrebleu.corpus_bleu(predictions, refs_for_bleu)
    #metrics["eval_ins_bleu"] = bleu_results.score

    # --- ROUGE-L ---
    rouge_calculator = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [rouge_calculator.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, single_refs)]
    metrics["eval_ins_rougeL"] = np.mean(rouge_l_scores) if rouge_l_scores else 0

    # --- METEOR ---
    meteor_scores = [meteor_score([ref.split()], pred.split()) for pred, ref in zip(predictions, single_refs)]
    metrics["eval_ins_meteor"] = np.mean(meteor_scores) if meteor_scores else 0
    
    # --- CIDEr ---
    cider_calculator = Cider()
    cider_score, _ = cider_calculator.compute_score(gts, res)
    metrics["eval_ins_cider"] = cider_score

    # --- SPICE ---
    spice_calculator = Spice()
    spice_score, _ = spice_calculator.compute_score(gts, res)
    metrics["eval_ins_spice"] = spice_score
    
    print("Finished calculating text metrics.")
    return metrics