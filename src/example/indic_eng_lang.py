import torch
import sys, os, pdb
import argparse, logging
import torch.nn.functional as F

from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'dialect'))

from mms_dialect import MMSWrapper
from whisper_dialect import WhisperWrapper


# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


if __name__ == '__main__':

    # pretrained_model = "whisper_large"
    pretrained_model = "mms-lid-256"

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
   
    indic_language_labels = [
        "assamese",
        "bengali",
        "bodo",
        "dogri",
        "english",
        "gujarati",
        "hindi",
        "kannada",
        "kashmiri",
        "konkani",
        "maithili",
        "malayalam",
        "manipuri",
        "marathi",
        "nepali",
        "odia",
        "punjabi",
        "sanskrit",
        "santali",
        "sindhi",
        "tamil",
        "telugu",
        "urdu"
    ]
    indic_language_labels.sort()

    # Define the model wrapper
    if pretrained_model == "mms-lid-256":
        log_dir = "/data2/kevinyhu/voxlect_weights/inla_english/mms-lid-256/lr00005_ep15_lora_64_accumulation_frozen"
        model = MMSWrapper(
            pretrain_model=pretrained_model, 
            finetune_method="lora",
            lora_rank=64, 
            output_class_num=len(indic_language_labels),
            freeze_params=True, 
            use_conv_output=True
        ).to(device)
    elif pretrained_model == "whisper_large":
        log_dir = "/data2/kevinyhu/voxlect_weights/inla_english/whisper_large/lr00005_ep15_lora_64_accumulation_frozen"
        model = WhisperWrapper(
            pretrain_model=pretrained_model, 
            finetune_method="lora",
            lora_rank=64, 
            output_class_num=len(indic_language_labels),
            freeze_params=True, 
            use_conv_output=True
        ).to(device)

    model.load_state_dict(torch.load(str(log_dir.joinpath(f'fold_1.pt')), weights_only=True), strict=False)
    model.load_state_dict(torch.load(str(log_dir.joinpath(f'fold_lora_1.pt'))), strict=False)
    model.eval()
    
    max_audio_length = 15 * 16000
    data = torch.zeros([1, 16000]).float().to(device)[:, :max_audio_length]
    logits, embeddings = model(data, return_feature=True)
        
    # Probability and output
    indic_language_prob = F.softmax(logits, dim=1)
    print(indic_language_labels[torch.argmax(indic_language_prob).detach().cpu().item()])
