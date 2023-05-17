from glob import glob
from sys import platform
import librosa
import math
from random import shuffle
import pickle
import os.path
import numpy as np


sample_rate = 22050   # sampling frequency
duration = 30         # length of the tracks in seconds

n_fft = 2048          # number of samples per fft - the size of the window when performing an fft
n_mfcc_mel = 128           # number of extracted coefficients
n_mfcc = 13
hop_length = 512      # the amount we shift with each fft

number_of_segments = 10      # the number of segments we want to split each track
samples_per_track = sample_rate * duration  # how many samples do we get from each track
samples_per_segment=int(samples_per_track/number_of_segments)    # how many samples are there in a segment
num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)   # this is to check if the output has the correct dimensions

def build_dataset(path, mode):
    audio_files = glob(path)

    if platform == "linux" or platform == "linux2":
        # linux
        separator = "/"
        sep_num = 2
    elif platform == "darwin":
        # OS X
        print("unlucky")
    elif platform == "win32":
        # Windows...
        separator = "\\"
        sep_num = 1

    data = {
        "mapping": [],  # mapping the names of the genres to indexes 0 to 9
        "mfcc": [],     # array containing the mfcc arrays of the track segments
        "labels": []    # array of the genre labels of the track segments
    }  

    #shuffle(audio_files)    # the audio files are ordered by category, its easier to shuffle them here while we only have to shuffle one array
    for audio_file in audio_files:
        # cutting the name of the genre from the filename
        genre = audio_file.split(separator)[sep_num]
        # adding genre to mapping if its not already there
        if genre not in data["mapping"]:      
            data["mapping"].append(genre) 
        try:
            # reading signal and sample rate from the file
            signal, sr = librosa.load(audio_file) 
        except:
            #there are some corrupted/non readable files so we dont process them
            continue
            
        # we dont have much data, so we split the tracks into segments to increase our training data
        for i in range(number_of_segments):
            # calculating start and finish index of the segment
            start = samples_per_segment * i
            end = start + samples_per_segment
            # Calculating the mfcc of the segment
            if mode == 'mfcc':
                feature = librosa.feature.mfcc(y=signal[start:end], sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                feature = feature.T.tolist()
            elif mode == 'mel':
                feature = librosa.feature.melspectrogram(y=signal[start:end], sr=sample_rate, n_mels=n_mfcc_mel, n_fft=n_fft, hop_length=hop_length)
                feature = (librosa.power_to_db(feature, ref=np.max) + 40) / 40
                feature = feature.T.tolist()

            # Some tracks are shorter than 30 seconds, so we have segments with incorrect length. We filter those out here
            if len(feature) == num_mfcc_vectors_per_segment:
                # Adding the mfcc and label to our data
                data["mfcc"].append(feature)
                data["labels"].append(data["mapping"].index(genre))
    
    if mode == 'mfcc':
        n = n_mfcc
    elif mode == 'mel':
        n = n_mfcc_mel
    with open('./Data/data_dict_'+mode+'_'+str(n)+'_'+str(number_of_segments)+'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_dataset(mode='mfcc'):
    if mode == 'mfcc':
        n = n_mfcc
        fname = './Data/data_dict_'+mode+'_'+str(n)+'_'+str(number_of_segments)+'.pickle'
        if not os.path.isfile(fname): 
            build_dataset('Data/genres_original/*/*.wav', mode='mfcc')
        with open(fname, 'rb') as handle:
            data = pickle.load(handle)
    elif mode == 'mel':
        n = n_mfcc_mel
        fname = './Data/data_dict_'+mode+'_'+str(n)+'_'+str(number_of_segments)+'.pickle'
        if not os.path.isfile(fname): 
            build_dataset('Data/genres_original/*/*.wav', mode='mel')
        with open(fname, 'rb') as handle:
            data = pickle.load(handle)
    else:
        print('Invalid mode')
        return None
    return data
