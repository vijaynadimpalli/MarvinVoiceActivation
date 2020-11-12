from DataCreator import *
from PycharmInferenceHelper import *
from RecordingClass import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pydub.playback import play
import time

model = load_model("best_model&weights.h5",custom_objects={'get_f1':get_f1,'f1_m':f1_m})
chime_file = "./chime.wav"

while(1):
  recorder(time_limit_sec=3)               #Simpler fixed length recorder.....
  #guiAUD = RecAUD(time_limit_sec=5)         #More sophisticated recorder with start and stop buttons....


  audio = AudioSegment.from_file('recording.wav', format="wav", frame_rate=48000)
  audio = audio.set_frame_rate(16000)

  pad_ms = 10000  # Add here the fix length you want (in milliseconds)
  if pad_ms > len(audio):
    silence = AudioSegment.silent(duration=pad_ms-len(audio),frame_rate=16000)
    audio = audio + silence  # Adding silence after the audio
  audio.export('recording.wav', format="wav")

  start_time = time.time()

  prediction = detect_triggerword('recording.wav',model)
  display_on_detect(prediction,0.5)

  print("Inference Time : %.2f sec"%(time.time()-start_time))

# chime_on_activate('recording.wav',chime_file, prediction, 0.5)
# out = AudioSegment.from_file('chime_output.wav')
# play(out)