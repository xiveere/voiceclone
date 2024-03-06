#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.configs.vits_config import VitsConfig


# In[2]:


output_path = os.path.dirname(os.path.abspath('__file__'))


dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="Mixed_formatted.txt",
    # meta_file_train = "No_Shouting_formatted.txt",
    path=os.path.join(output_path, "data/"),
    language = "en"
)


# In[3]:


audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)


# In[4]:


character_config = CharactersConfig(
    characters_class= "TTS.tts.models.vits.VitsCharacters",
    characters= "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890",
    punctuations=""" !,.?-"'""",
    pad= "<PAD>",
    eos= "<EOS>",
    bos= "<BOS>",
    blank= "<BLNK>",
)


# In[5]:


config = VitsConfig(
    audio=audio_config,
    # characters=character_config, # Comment out if with phonemes
    run_name="vits_tyler1_phonemes",
    # run_name = "vits_tyler1_noshouting_phonemes",
    batch_size=4,
    eval_batch_size=2,
    # batch_group_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=5000,
    text_cleaner="english_cleaners",
    use_phonemes=True, # Replace with False if no phonemes
    phoneme_language="en",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    phonemizer = 'espeak',
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path + "/vitstts_checkpoint",
    datasets=[dataset_config],
    cudnn_benchmark=False,
    test_sentences = ["So he starts off level one just doing this. Like oh he's gonna like bro he losing, doesn't hit a single thing. Look at it like- what the fuck. Like bro okay whatever. Watch this top dive bro. I solo made Vayne one HP, right? Just wait out and fucking ghost, you twat.", 
                      "Okay whatever you're auto-ing a ward sure it's fine. Bro what are you d- just wait you fucking freak. Where's he walking to by the way? What the fuck! It's not a win-trade this guy played as our Jarvan too. He's a one-trick like bro.",
                      "Hey! Sup? It's me, Tyler1 ready for the pre-alpha. We're back baby!"]
)


# In[6]:


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


# In[7]:


ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)


# In[8]:


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    # eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.1,
    formatter = formatter
)


# In[9]:


# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)


# In[10]:


# Make sure to change phoneme in config cell when training a phoneme model

model_path = "vitstts_checkpoint/vits_tyler1_phonemes-February-29-2024_09+00AM-e526ca1/"

trainer = Trainer(
    TrainerArgs(continue_path = model_path), # Load from checkpoint
    # TrainerArgs(restore_path = "vitstts_checkpoint/tts_models--en--ljspeech--vits/model_file.pth"),
    # TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)


# In[ ]:


trainer.fit()

