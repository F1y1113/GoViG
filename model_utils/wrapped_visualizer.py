import torch
import os
import torch.nn as nn

from typing import Optional, Literal, List, Tuple

from PIL import Image
import numpy as np
import torch.nn.functional as F

from transformers import ChameleonForConditionalGeneration

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList


def pairwise_euclidean_distance(tensor):
    # tensor: (token_num, embedding_dim)
    # Calculate squared norms of each row (token)
    squared_norms = torch.sum(tensor**2, dim=1, keepdim=True)  # (token_num, 1)

    # Use broadcasting to calculate pairwise squared Euclidean distances
    distances_squared = squared_norms + squared_norms.T - 2 * torch.matmul(tensor, tensor.T)

    # Due to possible floating-point precision issues, clamp to avoid negative values
    distances_squared = torch.clamp(distances_squared, min=0.0)

    # Calculate Euclidean distance
    distances = torch.sqrt(distances_squared)
    
    return distances

class AnoleforConditionalGeneration(ChameleonForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.image_decoder = None   # for having the image_decoder property, L516 in customize_trainer.py
        self.generate_with_embeds = False

        self.image_postprocess = True   # for postprocessing the pixel value with processor

        self.sketch_resolution = (self.model.vqmodel.config.resolution, self.model.vqmodel.config.resolution) # fixme
        
        self.image_token_num = 784

        self.bpe_indices = self.model.vocabulary_mapping.image_token_ids
        self.img_indices = [self.model.vocabulary_mapping.bpe2img[i] for i in self.bpe_indices]

        if "codebook_sim" in kwargs:
            self.codebook_sim = kwargs['codebook_sim']
        else:
            self.codebook_sim = None
    
    def get_vis_codebook_sim(self):
        if self.codebook_sim == "mse":
            self.codebook_sim_matrix = pairwise_euclidean_distance(self.model.vqmodel.quantize.embedding.weight.data.to(torch.float64)).to(torch.bfloat16)
        else:
            self.codebook_sim_matrix = None
    
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            multimodal_generation_mode: Optional[
                Literal["text-only", "image-only", "interleaved-text-image", "unrestricted"]
            ] = "interleaved-text-image",
            **kwargs,
    ):
        generate_ids = super().generate(
            inputs=inputs, 
            generation_config=generation_config, 
            logits_processor=logits_processor, 
            multimodal_generation_mode=multimodal_generation_mode,
            do_sample=True,
            **kwargs
        )

        if multimodal_generation_mode == "text-only":
            return generate_ids[:, kwargs["input_ids"].shape[-1]:], None
        
        elif multimodal_generation_mode == "image-only":
            response_ids = generate_ids[:, kwargs["input_ids"].shape[-1]:]
            return response_ids, None
        
        elif multimodal_generation_mode in ["interleaved-text-image", "unrestricted"]:
            response_ids = generate_ids[:, kwargs["input_ids"].shape[-1]:]
            return response_ids, None
    
    def recursive_generate(
        self,
        processor, 
        input_text, 
        save_dir,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        multimodal_generation_mode: Optional[
            Literal["text-only", "image-only", "interleaved-text-image", "unrestricted"]
        ] = "interleaved-text-image",
        **kwargs,
    ):
        """
        currently only support batch size = 1
        """
        max_try = 60
        init_img = kwargs['pixel_values'].to(self.device).to(torch.bfloat16)
        img_list = [self.model.get_image_tokens(img.unsqueeze(0)).to(torch.int64) for img in init_img]

        end_flag = False

        all_images = []
        all_pil_images = []
        all_images += img_list

        previous_text = input_text

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(max_try):
            if i == max_try-1:
                if "max_new_tokens" in kwargs:
                    kwargs['max_new_tokens'] = 300

            generated_result, _ = self.generate(
                inputs,
                generation_config,
                logits_processor,
                multimodal_generation_mode,
                **kwargs,
            )
            generated_results = split_token_sequence(
                tokens=torch.tensor(generated_result).to(self.model.device), 
                image_seq_length=self.image_token_num,
                boi=self.config.boi_token_id, 
                eoi=self.config.eoi_token_id,
                max_length=generated_result.shape[-1],
                pad_token_id=self.config.eos_token_id
            )
            
            pred_text = processor.batch_decode(generated_results['texts'], skip_special_tokens=True)[0]

            if generated_results["images"] is not None:
                generated_imgs = generated_results["images"][0].to(self.model.device)
                if "Carrying objects" in previous_text:
                    previous_text = previous_text.replace("<image>Carrying objects: None. ", "").replace("<image>Carrying objects: printer_0. ", "")
                    # add back init maze image
                    previous_text = previous_text.replace("Initial State: ", "Initial State: <image>Carrying objects: None. ")
                else:
                    previous_text = previous_text.replace("<image>", "")
                    # add back init maze image
                    previous_text = previous_text.replace("Initial maze: ", "Initial maze: <image>").replace("Initial State: ", "Initial State: <image>")
            else:
                generated_imgs = None

            if generated_imgs is not None:

                if len(pred_text.strip()) == 0:
                    updated_text = previous_text + "<image>"
                else:
                    updated_text = previous_text + "<image>" + pred_text.strip() + " "
                
                if len(img_list) == 2:
                    _ = img_list.pop(-1)
                img_list.append(generated_imgs)
                all_images.append(generated_imgs)

                img = self.decode_image_tokens(generated_imgs)
                img = processor.postprocess_pixel_values(img).squeeze()
                img = Image.fromarray(img.permute(1, 2, 0).detach().cpu().numpy())
                img.save(os.path.join(save_dir, f"{i}.jpg"))
                all_pil_images.append(img)
                
            else:
                updated_text = previous_text + pred_text.strip()
                updated_text += " "
            
            print(updated_text+"\n\n")
            
            # if "the answer is" in pred_text.lower():
            #     end_flag = True
            #     break

            if "stop" in pred_text.lower():
                end_flag = True
                break

            tokenized_input = processor(
                text=updated_text, 
                padding="max_length",
                return_tensors="pt",
                max_length=2600
            )

            tokenized_input = {k: v.to(self.device) for k, v in tokenized_input.items()}

            tokenized_input['input_ids'][tokenized_input['input_ids'] == self.model.config.image_token_id] = torch.cat(img_list).reshape(-1)
            
            kwargs['input_ids'] = tokenized_input['input_ids']
            kwargs['attention_mask'] = tokenized_input['attention_mask']

            previous_text = updated_text
            
        return pred_text, updated_text, all_images, all_pil_images



def split_token_sequence(
    tokens: torch.LongTensor,
    image_seq_length: int, 
    boi: int,
    eoi: int,
    max_length: int,
    pad_token_id: int
) -> List[Tuple[str, torch.LongTensor]]:
    """
    Split a sequence of tokens into text and image segments.
    
    Args:
        tokens (torch.LongTensor): The token sequence.
        boi (int): Begin of image token.
        eoi (int): End of image token.
    
    Returns:
        List[Tuple[str, torch.LongTensor]]: List of tuples indicating segment type and tokens.
    """
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    
    device = tokens.device
    tokens = tokens[0]  # remove batch dimension
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            # if entering an image segment, save the current text segment (if any)
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            # if exiting an image segment, save the current image segment
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    # save any remaining tokens
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))

    generated_imgs = []
    generated_texts = []
    for seg_id, (seg_type, seg_tokens) in enumerate(segments):
        if seg_type == "image_seg":
            assert seg_tokens.shape[1] == image_seq_length
            generated_imgs.append(seg_tokens)
        else:
            assert seg_type == "text_seg"
            generated_texts.append(seg_tokens.view(-1))

    text_tokens = torch.cat(generated_texts)
    if max_length > text_tokens.shape[-1]:
        text_tokens = torch.cat((text_tokens, torch.full((max_length-text_tokens.shape[-1],), fill_value=pad_token_id, device=text_tokens.device))).unsqueeze(0)
    elif max_length < text_tokens.shape[-1]:
        text_tokens = text_tokens.unsqueeze(0)[:, :max_length]
    else:
        text_tokens = text_tokens.unsqueeze(0)
    return {
        "texts": text_tokens,
        "images": generated_imgs if len(generated_imgs) != 0 else None
    }