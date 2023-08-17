# import neccessary libraries, install if absent via terminal
import time
print(f"Initializing Edge Device")
time.sleep(1)
print(f"Importing Libraries")
time.sleep(1)

import subprocess
import wave
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio

import os
from datetime import datetime
import requests

print(f"Done Initializing")

# Target classes for inference
class_names = {
    1: 'car-or-truck', 2: 'motorvehicle-horn', 3: 'boda-boda-motocyle', 4: 'motorvehicle-siren',
    5: 'car-alarm', 6: 'mobile-music', 7: 'hawker-vendor', 8: 'community-radio', 9: 'regilious-venue',
    10: 'herbalists', 11: 'construction-site', 12: 'fabrication-workshop', 13: 'generator',
    14: 'bar-restaurant-nightclub', 15: 'animal', 16: 'crowd-noise', 17: 'school', 18: 'street-preacher',
    0: 'other'
}

# Load the model from the directory
print(f"Loading Model Into Memory")
noiseclassifier_model = tf.saved_model.load("embedmodel")
print(f"Model Loaded successfully")

# Ensure the audio file is sampled correctly
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

#Credentials to interface with Sunbird Metrics API
device_id = 'NOISE-CLASSIFIER' 
file_name = 'inference_results.txt'
# Create a dictionary with the parameters to send
data = {
    "device": device_id
}
url = 'http://noise-sensors-dashboard.herokuapp.com/analysis/metrics-file/'

# Main loop that repeats every hour
print(f"--Settings--")
print(f"Recording and Inference - 10 Min Interval")
print(f"Uploading Inference File - 1 hour Interval")
print(f"Device ID- {device_id}")

#while True:
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Sequence started at: {timestamp}")
    
with open("inference_results.txt", "w") as results_file:
    for i in range(6):
        rtimestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_file.write(f"* Recording {i+1} - {rtimestamp}\n")
        arecord_command = f"arecord -D \"plughw:3,0\" -f S16_LE -r 16000 -d 5 -t wav output.wav"
        subprocess.run(arecord_command, shell=True)

        print(f"{i+1} Recorded at - {rtimestamp}")

        filename = 'output.wav'  # Change this to the recorded file's name
        audio = load_wav_16k_mono(filename)

        results = noiseclassifier_model(audio)
        your_top_class = np.argmax(results, axis=0)
        
        your_inferred_class = class_names[your_top_class]
        class_probabilities = tf.nn.softmax(results, axis=-1).numpy()

        results_file.write(f'Predicted class: {your_top_class} - {your_inferred_class} (Probability: {class_probabilities[your_top_class]:.4f})\n')
        
        # Print the inferred class and its probability
        print(f'[Your model] The main noise is: {your_inferred_class} ({class_probabilities[your_top_class]})')

        dest = f"inference/{i+1}-{your_inferred_class}-{class_probabilities[your_top_class]:.4f}.wav"
        src = filename
        results_file.write(dest + "\n")
        os.rename(src, dest)

        for j, class_prob in enumerate(class_probabilities):
            class_namex = class_names[j]
            results_file.write(f'Class: {class_namex}, Probability: {class_prob:.4f}\n')

        
        # time.sleep(600)  # Sleep for 10 minutes between recordings
        time.sleep(10)  # Sleep for 1 minutes between recordings


    # Send the file once an hour
    #if datetime.now().minute == 0:  # Send at the start of each hour
with open(file_name, "rb") as file:
    files = {"metrics_file": (file_name, file)}  # Use the correct field name

    response = requests.post(url, data=data, files=files)

    if response.status_code == 201:
        print("File sent successfully")
        print("Response status code:", response.status_code)
        print("Response content:", response.content)
    else:
        print("Failed to send file")
        print("Response status code:", response.status_code)
        print("Response content:", response.content)

# Sleep for a minute before the next iteration
time.sleep(60)  # 60 seconds = 1 minute
