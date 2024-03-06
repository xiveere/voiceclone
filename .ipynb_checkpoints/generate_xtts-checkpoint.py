#!/usr/bin/env python
# coding: utf-8

# # Generate

# ### Init model

# In[1]:


import os
import torch
import torchaudio
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
from subprocess import getoutput
from IPython.display import Audio
from flask import Flask, request, jsonify
import logging
import time
import boto3
from botocore.exceptions import NoCredentialsError
from TTS.tts.models import setup_model as setup_tts_model
from TTS.config import load_config
import time

logging.basicConfig(filename = "app.log", level = logging.INFO, format='%(asctime)s%(levelname)s:%(message)s')
logging.info("Starting model init")
start = time.time()
app = Flask(__name__)

# Add here the xtts_config path
CONFIG_PATH = "xttsv2_checkpoint/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "xttsv2_checkpoint/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "xttsv2_checkpoint/best_model.pth"

# List of all wavs for speaker reference
wavs = getoutput("ls data/wavs/*.wav").split("\n")
# Add here the speaker reference
SPEAKER_REFERENCE = ["data/wavs/" + wav for wav in os.listdir('data/wavs/') if "wav" in wav]

# config = XttsConfig()
# config.load_json(CONFIG_PATH)
# model = Xtts.init_from_config(config)
config = load_config(CONFIG_PATH)
model = setup_tts_model(config)
model.load_checkpoint(config, checkpoint_dir = "xttsv2_checkpoint/",
                      checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.to("cuda")

gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path= SPEAKER_REFERENCE)
end = time.time()
runtime = end - start

logging.info(f"Model init complete in {runtime}s")


# ### S3 push

# In[2]:


bucket_name = "audio-messages-bucket"
url = "https://d1d78cjctwypjb.cloudfront.net/"

def push_to_s3(file_path, bucket_name, s3_key):
    """
    Upload a file to an S3 bucket

    :param file_path: Path to file to upload
    :param bucket_name: Name of the bucket to upload to
    :param s3_key: S3 object name. If not specified, file_path is used
    :return: True if file was uploaded, else False
    """
    
    # Create an S3 client
    s3_client = boto3.client('s3')
    try:
        # Upload the file
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logging.info(f"File {file_path} uploaded to {bucket_name}/{s3_key}.")
        return True
    except NoCredentialsError:
        logging.error("Credentials not available or invalid.")
        return False
    except Exception as e:
        logging.error(f"Failed to upload file: {e}")
        return False


# ### Supabase push

# In[ ]:


import os
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

def push_to_supabase(user_id, creator, game_id, timestamp, link):

    try:
        load_dotenv()
    
        database_url = os.getenv("DATABASE_URL")

        logging.info("Connecting to database")
        conn = psycopg2.connect(database_url)
    
        table_name = "message_generation",
        cur = conn.cursor()

        logging.info("Writing to database")
        sql = """
        UPDATE message_generation
        SET audio = %s
        WHERE id = %s
        """
        
        data = (link, id)
        
        # Execute the command and pass in the data
        cur.execute(sql, data)

        
        
        conn.commit()
        logging.info(f"Audio link written to {table_name} in id = {id}")
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logging.error(f"Failed to write to database: {str(e)}")
        return False


# ### Output

# In[3]:


@app.route('/inference', methods = ['POST'])
def inference():
    try:
        request_data = request.json

        # Model info
        text = request_data['text']
        language = request_data['language']
        creator = request_data['creator']

        # User info
        user_id = request_data['user_id']
        game_id = request_data['game_id']
        timestamp = request_data['timestamp']
        
    except Exception as e:
        logging.error(f"Missing data: {str(e)}")
        return jsonify({'error': 'Missing data in request',
                        'message' : str(e)})

    
    key = "/".join(user_id,language,creator,game_id,timestamp)
    link = url + key
    logging.info(f"Request received for {key}. Starting inference")
    start = time.time()
    
    try:
        out = model.inference(text, language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7, # Add custom parameters here
        )

        out_path = "output.wav"
        torchaudio.save(out_path, torch.tensor(out["wav"]).unsqueeze(0), 22050)

    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return jsonify({'error' : 'Error during inference')
    
    end = time.time()
    logging.info(f"Audio generated in {end}s")

    logging.info(f"Pushing to S3 bucket")
    push_result = push_to_s3(out_path, bucket_name, link)

    if push_result:
        logging.info(f"Pushed to {link}")

    else:
        return jsonify({'error': 'Error during push to S3'})

    push_result = push_to_supabase(user_id, creator, game_id, timestamp, link)

    if push_result:
        logging.info(f"Pushed to {link}")
        return jsonify({'audio_url' : link,
                        'run_time' : end-start})
    else:
        return jsonify({'error': 'Error during push to Supabase'})


# In[4]:

if __name__ == '__main__'
app.run(debug = True)

