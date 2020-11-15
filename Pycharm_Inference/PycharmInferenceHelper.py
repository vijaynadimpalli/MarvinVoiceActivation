import tensorflow as tf
from tensorflow.keras import backend as K
import pyaudio
import wave
#import keyboard
from build.DataCreator import *
import webbrowser

def detect_triggerword(filename, model):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    # plt.subplot(2, 1, 2)
    # plt.plot(predictions[0, :, 0])
    # plt.ylabel('probability')
    # plt.show()
    return predictions


def chime_on_activate(filename, chime_file, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            consecutive_timesteps = 0

    audio_clip.export("chime_output.wav", format='wav')


def display_on_detect(predictions,threshold):
    Ty = predictions.shape[1]
    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 10:
            print("ACTIVATED!!!!")
            webbrowser.open('google.com')
            break




def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Another F1 function implementation(Using inbuilt metrics).....
pres = tf.keras.metrics.Precision()
rec = tf.keras.metrics.Recall()

def get_f1(y_true,y_pred):

  #pres.reset_states()             #Using these reset_states gives same values as given by f1_m function
  #rec.reset_states()              #reset_states clears any persistence,and clears it for every batch,giving same functionality as f1_m
  pres.update_state(y_true,y_pred) #Not using reset_states averages out the f1 values over all batches in an epoch and doesn't reset with new epoch...
  rec.update_state(y_true,y_pred)

  return 2*pres.result()*rec.result()/(pres.result()+rec.result()+K.epsilon())

def recorder(time_limit_sec=10):
    # the file name output you want to record into
    filename = "recording.wav"
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 44100 samples per second
    sample_rate = 48000
    record_seconds = time_limit_sec
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
        #if keyboard.is_pressed('q'):
        #    break
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(filename, "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()
