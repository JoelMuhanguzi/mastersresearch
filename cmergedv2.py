import subprocess
#import pyaudio
import wave
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import time
import os

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

for i in range(10):
    print(f"* recording {i+1} ")
    arecord_command = f"arecord -D \"plughw:3,0\" -f S16_LE -r 16000 -d 10 -t wav output.wav"
    subprocess.run(arecord_command, shell=True)
    print("* done recording")

    # Load the recorded audio and perform inference
    filename = 'output.wav'  # Change this to the recorded file's name
    audio = load_wav_16k_mono(filename)

    results = yamnet_model(audio)
    class_ = np.argmax(results, axis=0)

    prediction = class_names[class_]
    print(f"The predicted class is {class_} and that is {prediction}")

    # Determine the class with the highest probability
    your_top_class = class_
    your_inferred_class = class_names[your_top_class]
    
    # Calculate class probabilities using softmax and convert to NumPy array
    class_probabilities = tf.nn.softmax(results, axis=-1).numpy()

    # Print the inferred class and its probability
    print(f'[Your model] The main sound is: {your_inferred_class} ({class_probabilities[your_top_class]})')
    dest = "14AugInference/" + f"{i+1}-" + your_inferred_class + f"-{class_probabilities[your_top_class]:.4f}.wav"
    src ='output.wav'
    print(dest)
    os.rename(src,dest)
    # Print probabilities of all classes
    for i, class_prob in enumerate(class_probabilities):
        class_namex = class_names[i]
        #print(f'Class: {class_namex}, Probability: {class_prob:.4f}')
    
    time.sleep(10)
    
