import subprocess
import wave
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import time
import os
from datetime import datetime
import requests

class_names = {
    1: 'car-or-truck', 2: 'motorvehicle-horn', 3: 'boda-boda-motocyle', 4: 'motorvehicle-siren',
    5: 'car-alarm', 6: 'mobile-music', 7: 'hawker-vendor', 8: 'community-radio', 9: 'regilious-venue',
    10: 'herbalists', 11: 'construction-site', 12: 'fabrication-workshop', 13: 'generator',
    14: 'bar-restaurant-nightclub', 15: 'animal', 16: 'crowd-noise', 17: 'school', 18: 'street-preacher',
    0: 'other'
}
yamnet_model = tf.saved_model.load("embedmodel")

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

device_id = 'SB1001'  # Replace with the actual device IDx
file_name = 'inference_results.txt'
# Create a dictionary with the parameters to send
data = {
    "device": device_id
}
url = 'http://noise-sensors-dashboard.herokuapp.com/analysis/metrics-file/'

with open("inference_results.txt", "w") as results_file:
    for i in range(3):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_file.write(f"* Recording {i+1} - {timestamp}\n")
        arecord_command = f"arecord -D \"plughw:3,0\" -f S16_LE -r 16000 -d 10 -t wav output.wav"
        subprocess.run(arecord_command, shell=True)
        results_file.write("* Done recording\n")

        filename = 'output.wav'  # Change this to the recorded file's name
        audio = load_wav_16k_mono(filename)

        results = yamnet_model(audio)
        class_ = np.argmax(results, axis=0)

        prediction = class_names[class_]
        results_file.write(f"Predicted class: {class_} - {prediction}\n")

        your_top_class = class_
        your_inferred_class = class_names[your_top_class]
        class_probabilities = tf.nn.softmax(results, axis=-1).numpy()

        results_file.write(f'Main sound: {your_inferred_class} (Probability: {class_probabilities[your_top_class]:.4f})\n')

        dest = f"14AugInference/{i+1}-{your_inferred_class}-{class_probabilities[your_top_class]:.4f}.wav"
        src = 'output.wav'
        results_file.write(dest + "\n")
        os.rename(src, dest)

        for j, class_prob in enumerate(class_probabilities):
            class_namex = class_names[j]
            results_file.write(f'Class: {class_namex}, Probability: {class_prob:.4f}\n')

        time.sleep(10)

# Send the text file and device ID to the server endpoint
# server_endpoint = 'http://noise-sensors-dashboard.herokuapp.com/analysis/metrics-file/"'  # Replace with the actual server endpoint URL
# files = {'file': ('inference_results.txt', open('inference_results.txt', 'rb')), 'device_id': (None, device_id)}
# response = requests.post(server_endpoint, files=files)
# 
# if response.status_code == 200:
#     print("File uploaded successfully.")
# else:
#     print("Error uploading file.")
    
# Open the file and send it as part of the request with the correct field name
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
