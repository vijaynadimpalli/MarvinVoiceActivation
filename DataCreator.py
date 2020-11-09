import wget
import os

import base64
import matplotlib as mpl
import tarfile


import random
from shutil import copyfile,make_archive

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram,get_window
import gc
from pydub import AudioSegment

import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import glob
import h5py

Tx = 1998  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram
Ty = 496  # The number of time steps in the output of our model


def create_onedrive_directdownload(onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl



# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file,is_train=False):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim

    #Values from spectrogram and plt.specgram are not matching..... so only use specgram for now


    # if is_train:
    #   if nchannels == 1:
    #     freqs, bins, pxx = spectrogram(data, fs,window=get_window('hann',nfft),nperseg=nfft, noverlap = noverlap,scaling='spectrum')
    #   elif nchannels == 2:
    #     freqs, bins, pxx = spectrogram(data[:,0], fs,window=get_window('hann',nfft),nperseg=nfft, noverlap = noverlap,scaling='spectrum')
    
    # else:
    #   if nchannels == 1:
    #     pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    #   elif nchannels == 2:
    #     pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)

    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)

    #plt.savefig('train.png')
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    #plt.close()

    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(silent_background=False):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)            
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav") and filename=='exercise_bike.wav': #Only considering one background for now
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)

    if silent_background:
      backgrounds = [AudioSegment.silent(duration=10001,frame_rate=16000)]
    return activates, negatives, backgrounds




def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0,
                                      high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)



def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = overlap | True
    ### END CODE HERE ###

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    ### START CODE HERE ###
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)
    ### END CODE HERE ###

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time



def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < y.shape[1]:
            y[0, i] = 1
    ### END CODE HERE ###

    return y


def create_training_example(backgrounds, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Set the random seed
    #np.random.seed(18)


    background = backgrounds[np.random.randint(0,len(backgrounds))]
    #Get random 10sec segment
    segment_start = np.random.randint(low=0,
                                      high=len(background) - 10000)

    background = background[segment_start:segment_start + 10000]


    background = background - 20

    ### START CODE HERE ###
    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1, Ty))

    # Step 2: Initialize segment times as an empty list (≈ 1 line)
    previous_segments = []
    ### END CODE HERE ###

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    ### START CODE HERE ### (≈ 3 lines)
    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end)
    ### END CODE HERE ###

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    ### START CODE HERE ### (≈ 2 lines)
    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    ### END CODE HERE ###

    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)
    file_handle = background.export("train" + ".wav", format="wav")
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")

    return x.T, y.T

    
def main(args):
    
    num_actives, num_negatives, train_examples, val_examples, chunk_size, silent_background = args
    
    wget.download('http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz')
    os.mkdir('./raw_data')
    os.mkdir('./raw_data/negatives')
    os.mkdir('./raw_data/activates')
    os.mkdir('./raw_data/backgrounds')

    os.mkdir('./XY_train')
    os.mkdir('./XY_dev')
    
    #getting chime.wav

    wget.download(create_onedrive_directdownload('https://1drv.ms/u/s!AjYYbRcfzZT5yxF7J6tKXP5yp8vl?e=psvSJK'))
    
    backend_ =  mpl.get_backend() 
    mpl.use("Agg")  # Prevent showing stuff
    
    file=tarfile.open('/content/speech_commands_v0.01.tar.gz')

    file.extractall(path='./data')
    
    activates_folder = os.path.join('./data','marvin')
    backgrounds_folder =  os.path.join('./data','_background_noise_')    
    
    files = [os.path.join('./data',x) for x in os.listdir('./data')]
    files.remove(activates_folder)
    files.remove(backgrounds_folder)

    folders = list(filter(lambda x: os.path.isdir(x),files))

    files_in = []
    for folder in folders:
      files_in.extend([os.path.join(folder,x) for x in os.listdir(folder)])


    for file in random.sample(files_in,num_negatives):
      copyfile(file,os.path.join('./raw_data/negatives',os.path.split(file)[-1]))

    for file in random.sample(os.listdir(activates_folder),num_actives):
      copyfile(os.path.join(activates_folder,file),os.path.join('./raw_data/activates',os.path.split(file)[-1]))

    for file in os.listdir(backgrounds_folder):
      copyfile(os.path.join(backgrounds_folder,file),os.path.join('./raw_data/backgrounds',os.path.split(file)[-1]))

    print(len(os.listdir('./raw_data/activates')))
    
    
    #************************************************************************************************************#
    
    plt.ioff()
    fig = plt.figure()
    # Load audio segments using pydub
    activates, negatives, backgrounds = load_raw_audio(bool(silent_background))

    f1 = h5py.File("./XY_train/XY.h5",'w')

    X_train = f1.create_dataset("X_train",(train_examples,Tx,n_freq),chunks=(chunk_size,Tx,n_freq))
    Y_train = f1.create_dataset("Y_train",(train_examples,Ty,1),chunks=(chunk_size,Ty,1))


    X = np.zeros((chunk_size,Tx,n_freq))
    Y = np.zeros((chunk_size,Ty,1))

    for i in np.arange(0,train_examples,chunk_size):
      for j in range(chunk_size):
        X[j], Y[j] = create_training_example(backgrounds, activates, negatives)

      print(str(i/chunk_size) + " Chunk stored")
      X_train[i:i+chunk_size,:,:] = X
      Y_train[i:i+chunk_size,:,:] = Y

    f1.close()


    f2 = h5py.File("./XY_dev/XY_dev.h5",'w')

    X_test = f2.create_dataset("X_test",(val_examples,Tx,n_freq),chunks=(chunk_size,Tx,n_freq))
    Y_test = f2.create_dataset("Y_test",(val_examples,Ty,1),chunks=(chunk_size,Ty,1))

    for i in np.arange(0,val_examples,chunk_size):
      for j in range(chunk_size):
        X[j], Y[j] = create_training_example(backgrounds, activates, negatives)

      print(str(i/chunk_size) + " Chunk stored")
      X_test[i:i+chunk_size,:,:] = X
      Y_test[i:i+chunk_size,:,:] = Y

    f2.close()


    #gc.collect()
    
if __name__ == "__main__":
    args = list(map(int,sys.argv[1:]))
    main(args)





