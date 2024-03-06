#!/usr/bin/env python
# coding: utf-8

# ## Train

# In[1]:


import torch
torch.cuda.empty_cache()

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# In[2]:


import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from TTS.config import load_config


# In[3]:


OUT_PATH = os.path.dirname(os.path.abspath("__file__"))

config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="Mixed_formatted.txt",
    # meta_file_train = "No_Shouting_formatted.txt",
    path=os.path.join(OUT_PATH, "data/"),
    language = "en"
)
# Define here the dataset that you want to use for the fine-tuning on.
# config_dataset = BaseDatasetConfig(
#     formatter="ljspeech",
#     dataset_name="ljspeech",
#     path="/raid/datasets/LJSpeech-1.1_24khz/",
#     meta_file_train="/raid/datasets/LJSpeech-1.1_24khz/metadata.csv",
#     language="en",
# )

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]


# In[4]:


# define audio config
audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=22050)
# training parameters config


# In[5]:


# Define the path where XTTS v1.1.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "xttsv2_checkpoint", "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )


# In[6]:


# init args and config
XXTS_CHECKPOINT = "xttsv2_checkpoint/XTTS_v2.0_original_model_files/model.pth"

# XTTS_CHECKPOINT = "xttsv2_checkpoint/tyler1_xttsv2-February-28-2024_01+16PM-e526ca1/best_model.pth"

model_args = GPTArgs(
    max_conditioning_length=int(132300*2),  # 12 secs
    min_conditioning_length=66150,  # 3 secs
    debug_loading_failures=True,
    max_wav_length=255995*3,  # ~33 seconds
    max_text_length=700,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
    # gpt_max_text_tokens = 500,
    # gpt_max_prompt_tokens = 100
)


# In[7]:


# Training sentences generations
# SPEAKER_REFERENCE = [
#     "data/wavs/1. Tyler1 THE_WORST_JUNGLER_EVER 1.wav"  # speaker reference to be used in training test sentences
# ]

SPEAKER_REFERENCE = ["data/wavs/" + wav for wav in os.listdir('data/wavs/') if "wav" in wav]

LANGUAGE = config_dataset.language

config = GPTTrainerConfig(
    output_path=OUT_PATH + '/xttsv2_checkpoint',
    model_args=model_args,
    run_name="tyler1_xttsv2",
    project_name="tyler1",
    dashboard_logger="tensorboard",
    # logger_uri=None,
    audio=audio_config,
    batch_size=1,
    # batch_group_size=48,
    eval_batch_size=1,
    num_loader_workers=4,
    # eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=10000,
    save_n_checkpoints=1,
    save_checkpoints=True,
    # target_loss="loss",
    print_eval=True,
    # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
    optimizer="AdamW",
    optimizer_wd_only_on_weights=True, # for multi-gpu training please make it False
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,  # learning rate
    lr_scheduler="MultiStepLR",
    # it was adjusted accordly for the new step scheme
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[
        {
            "text": "So he starts off level one just doing this. Like oh he's gonna like bro he losing, doesn't hit a single thing. Look at it like- what the fuck. Like bro okay whatever. Watch this top dive bro. I solo made Vayne one HP, right? Just wait out and fucking ghost, you twat.",
            "speaker_wav": SPEAKER_REFERENCE,
            "language": LANGUAGE,
        },
        {
            "text": "Okay whatever you're auto-ing a ward sure it's fine. Bro what are you d- just wait you fucking freak. Where's he walking to by the way? What the fuck! It's not a win-trade this guy played as our Jarvan too. He's a one-trick like bro.",
            "speaker_wav": SPEAKER_REFERENCE,
            "language": LANGUAGE,
        },
        {
            "text": "Hey! Sup? It's me, Tyler1 ready for the pre-alpha. We're back baby!",
            "speaker_wav": SPEAKER_REFERENCE,
            "language": LANGUAGE,
        },
    ],
    # mixed_precision = False
)


# In[8]:


def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "Tyler1"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.dirname(os.path.abspath('__file__')) + f"/data/wavs/{cols[0]}.wav"
            text = cols[1]
            # print(text)
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items


train_samples, eval_samples = load_tts_samples(
    config_dataset,
    # DATASETS_CONFIG_LIST,
    eval_split=True,
    # eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.1,
    formatter = formatter
)


# In[9]:


# init the model from config
model = GPTTrainer.init_from_config(config)


# In[10]:


# init the trainer and 🚀

# model_path = "xttsv2_checkpoint/tyler1_xttsv2-February-19-2024_05+30AM-2b31060/"
trainer = Trainer(
    TrainerArgs(
        # continue_path = model_path,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=256, # batch_size * grad_accum_steps >= 256,
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)


# In[ ]:


trainer.fit() 