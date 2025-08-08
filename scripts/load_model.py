import requests
import torch
import math

# from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# from model_utils.wrapped_visualizer import KosmosVforConditionalGeneration

def load_model(args):
    print("image_seq_length received:", args.image_seq_length)
    model_name = args.model

    model_ckpt_path = args.model_ckpt

    if model_name in ['anole']:
        image_token_num = args.image_seq_length

        from model_utils.wrapped_visualizer import AnoleforConditionalGeneration
        model = AnoleforConditionalGeneration.from_pretrained(
            "leloy/Anole-7b-v0.1-hf",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            codebook_sim="mse"
        )
        processor = AutoProcessor.from_pretrained("leloy/Anole-7b-v0.1-hf", image_seq_length=image_token_num)

        def generate_bin_tokens(prefix, vmin, vmax, step):
            nbins = int((vmax - vmin) / step) + 1
            return [f"<{prefix}_bin_{i:02d}>" for i in range(nbins)]

        bin_tokens = []
        bin_tokens += generate_bin_tokens("dx", -0.18, 0.18, 0.01)
        bin_tokens += generate_bin_tokens("dy", -0.18, 0.18, 0.01)
        bin_tokens += generate_bin_tokens("dyaw", -0.4, 0.4, 0.01)

        existing_vocab = set(processor.tokenizer.get_vocab().keys())
        new_tokens = [t for t in bin_tokens if t not in existing_vocab]

        if new_tokens:
            processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
            
            # Handle PEFT model token embedding resizing
            # Check for both possible PEFT wrapper types
            if (hasattr(model.lm_head, 'base_layer') or 
                hasattr(model.lm_head, '__class__') and 
                'ModulesToSaveWrapper' in str(model.lm_head.__class__)):
                from peft.utils.other import ModulesToSaveWrapper
                # Unwrap the lm_head
                original_lm_head = model.lm_head
                if hasattr(model.lm_head, 'base_layer') or 'ModulesToSaveWrapper' in str(type(model.lm_head)):
                    # PEFT model - unwrap, resize, and re-wrap
                    if hasattr(model.lm_head, 'base_layer'):
                        model.lm_head = model.lm_head.base_layer
                    elif hasattr(model.lm_head, 'original_module'):
                        model.lm_head = model.lm_head.original_module
                    
                    # Resize embeddings
                    model.resize_token_embeddings(len(processor.tokenizer))
                    
                    # Re-wrap the lm_head with the correct adapter_name
                    model.lm_head = ModulesToSaveWrapper(model.lm_head, "default")
                else:
                    # Standard model without PEFT
                    model.resize_token_embeddings(len(processor.tokenizer))
            else:
                # Standard model without PEFT
                model.resize_token_embeddings(len(processor.tokenizer))

        # model.resize_token_embeddings(len(processor.tokenizer))
        # processor.image_processor.size = {"shortest_edge": int(512 / (math.sqrt(1024 / image_token_num)))}
        # processor.image_processor.crop_size = {
        #     "height": int(512 / (math.sqrt(1024 / image_token_num))),
        #     "width": int(512 / (math.sqrt(1024 / image_token_num)))
        # }
        # processor.image_processor.size = {"shortest_edge": int(384 / int(math.sqrt(768 / image_token_num)))}
        # processor.image_processor.crop_size = {
        #     "height": int(384 / int(math.sqrt(768 / image_token_num))),
        #     "width": int(384 / int(math.sqrt(768 / image_token_num)))
        # }
        # Calculate image size based on desired token count (729 = 27×27)
        processor.image_processor.size = {"shortest_edge": 448}  # This is an estimate, may need adjustment
        processor.image_processor.crop_size = {
            "height": 448,
            "width": 448
        }

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
        model.model.vqmodel.config.resolution = processor.image_processor.size["shortest_edge"]
        model.model.vqmodel.quantize.quant_state_dims = [
            model.model.vqmodel.config.resolution // 2 ** (len(model.model.vqmodel.config.channel_multiplier) - 1)
        ] * 2

        args.sketch_resolution = model.model.vqmodel.config.resolution
        model.sketch_resolution = (args.sketch_resolution, args.sketch_resolution)
        model.image_token_num = image_token_num

        model.get_vis_codebook_sim()

        from peft import LoraConfig, get_peft_model
        from peft.peft_model import PeftModel

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_proj', "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["lm_head"],
        )
        lora_model = get_peft_model(model, config)

        if (args.do_eval or args.do_predict) and not args.do_train and model_ckpt_path:
            lora_model = PeftModel.from_pretrained(model, model_ckpt_path, is_trainable=False)

        return {
            'processor': processor,
            'model': lora_model
        }
    else:
        raise ValueError("Unsupported model type. ")