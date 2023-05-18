from tensorflow import expand_dims
import numpy as np
import pytube
import librosa
import math
import os
import warnings
warnings.filterwarnings('ignore')

### hard-coded values, cba to make it a config file
sample_rate = 22050   # sampling frequency
duration = 30 

n_fft = 2048          # number of samples per fft - the size of the window when performing an fft
n_mfcc_mel = 128           # number of extracted coefficients
n_mfcc = 13
hop_length = 512      # the amount we shift with each fft

number_of_segments = 10      # the number of segments we want to split each track
samples_per_track = sample_rate * duration  # how many samples do we get from each track
samples_per_segment=int(samples_per_track/number_of_segments)    # how many samples are there in a segment
num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

def predict_genre(model, mapping, signal, sr=22050, threshold=0.7):
    segments = []
    n_segments = signal.shape[0] // samples_per_segment
    intro = 0
    outro = n_segments
    if n_segments > 20:
        intro = int(n_segments * 0.2)
        outro = n_segments - int(n_segments * 0.2)
    for i in range(intro, outro):
        # calculating start and finish index of the segment
        start = samples_per_segment * i
        end = start + samples_per_segment
        # Calculating the mfcc of the segment
        feature = librosa.feature.melspectrogram(y=signal[start:end], sr=sr, n_mels=n_mfcc_mel, n_fft=n_fft, hop_length=hop_length)
        feature = (librosa.power_to_db(feature, ref=np.max) + 40) / 40
        feature = feature.T.tolist()
        if len(feature) == num_mfcc_vectors_per_segment:
            # Adding the mfcc and label to our data
            segments.append(feature)

    segments = np.array(segments)
    aggr = np.zeros(10)
    for segment in segments:
        segment = expand_dims(segment, axis=-1)
        segment = expand_dims(segment, axis=0)
        p = model.predict(segment, verbose=0)
        if max(p[0]) > threshold:
            aggr += p[0] / np.sum(p[0])
    pred = np.argmax(aggr)

    #print(mapping[pred] + "\t" + str(aggr[np.argmax(aggr)] * 100) + "%")
    return mapping[pred]

def predict_from_youtube_link(model, mapping, link):
    yt = pytube.YouTube(link)
    title = yt.title
    
  
    # check for destination to save file
    destination = './demo/'
    
    # download the file
    title = title.replace(':', '').replace('?', '').replace('|', '').replace('"', '').replace('<', '').replace('>', '').replace('*', '').replace('\\', '').replace(',', '').replace('/', '').replace('.', '')
    outname = destination+title+'.mp3'
    print(outname)
    if not os.path.isfile(outname):
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=destination)
    
        # save the file
        base, ext = os.path.splitext(out_file)
        outname = base + '.mp3'
        os.rename(out_file, outname)
        print(outname)
    
    # result of success
    signal, sr = librosa.load(outname)
    
    r =  predict_genre(model, mapping, signal, sr)
    #os.remove(new_file)
    return r, title
    
