import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import python_speech_features as psf

Ty = 1496 # The number of time steps in the output of our model


list_person=np.arange(1, 73)
string_file="./data2/"
list_label1=["/1_label1.wav", "/2_label1.wav", "/3_label1.wav","/4_label1.wav" ,"/5_label1.wav","/6_label1.wav" ,"/7_label1.wav", "/8_label1.wav", "/9_label1.wav", "/10_label1.wav"]
list_label0=["/1_label0.wav", "/2_label0.wav", "/3_label0.wav","/4_label0.wav" ,"/5_label0.wav"]

# BGN section
BGN_Cafe_tmp = wavfile.read("./data/Cafeteria_10s.wav")
BGN_Pub_tmp = wavfile.read("./data/Pub_10s.wav")
BGN_Cafe = np.copy(BGN_Cafe_tmp)
BGN_Pub = np.copy(BGN_Pub_tmp)

# scale the background noise
tmp = np.copy(np.array(BGN_Cafe_tmp[1], dtype=np.float32))
tmp2 = np.round(32767*tmp)
BGN_Cafe[1] = tmp2.astype(np.int16)
BGN_Cafe[1] = BGN_Cafe[1]-100

tmp3 = np.copy(np.array(BGN_Pub_tmp[1], dtype=np.float32))
tmp4 = np.round(32767*tmp3)
BGN_Pub[1] = tmp4.astype(np.int16)
BGN_Pub[1] = BGN_Pub[1]


dic_person_label1={}
dic_person_label0={}
for i in list_person:
    dic_person_label1[i]=[]
    dic_person_label0[i]=[]

for i in list_person:
    for j in list_label1:
        data1 = wavfile.read(string_file + str(i) + j)
        dic_person_label1[i].append(data1)    
    for k in list_label0:
        data2 = wavfile.read(string_file + str(i) + k)
        dic_person_label0[i].append(data2)
    



def get_random_time_segment(segment_ms):

    segment_start = np.random.randint(low=0, high=480000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    
    segment_start, segment_end = segment_time

    overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip[1])
    segment_time = get_random_time_segment(segment_ms)
    
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)
        print("overlap")

    previous_segments.append(segment_time)

    new_background=background.copy()
    for i in range(len(audio_clip[1])):
        new_background[1][segment_time[0]+i] = new_background[1][segment_time[0]+i]+audio_clip[1][i]
    
    
    return new_background, segment_time

def insert_ones(y, segment_end_ms):
    
    # change the time into mini seconds, original 48000 per second
    segment_end_ms = (segment_end_ms/48000)*1000
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    for i in range(segment_end_y+1, segment_end_y+101):
        if i < Ty:
            y[0, i] = 1.0
    
    return y

def graph_spectrogram(wav_file):
  # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.specgram.html
  # FFT

  rate, data = wavfile.read(wav_file)
  nfft = 200 # Length of each window segment
  fs = 8000 # Sampling frequencies
  noverlap = 120 # Overlap between windows
  nchannels = data.ndim
  if nchannels == 1:
      pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
  elif nchannels == 2:
      pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
  return pxx


def create_training_example(background, activates, negatives, person, track):
    
    y = np.zeros((1, Ty))
    background_new = background.copy()
    previous_segments = []
    
    # randomly insert (0, 4)
    number_of_activates = np.random.randint(0, 4)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    for random_activate in random_activates:
        background_new, segment_time = insert_audio_clip(background_new, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)
        
    # randomly insert (1, 2)
    number_of_negatives = np.random.randint(1, 2)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]
    

    for random_negative in random_negatives:
        background_new, _ = insert_audio_clip(background_new, random_negative, previous_segments)

    # Export new training example
    wavfile.write(str(person)+"_"+str(track)+"_train.wav", background_new[0], background_new[1]) 
    np.savetxt(str(person)+"_"+str(track)+"_y.txt", y) 
    
    # feature extraction
    x = graph_spectrogram(str(person)+"_"+str(track)+"_train.wav")

    return x, y


def MFCC_feature_extraction_psf(wav_file):
    # Mel-frequency cepstral coefficients
    # DFT
    # https://jonathan-hui.medium.com/speech-recognition-feature-extraction-mfcc-plp-5455f5a69dd9
    # https://python-speech-features.readthedocs.io/en/latest/
    fs, sig = wavfile.read(wav_file)
    
    #mfccs = psf.mfcc(sig, samplerate=16000)
    #mfccs = psf.mfcc(sig, samplerate=fs, nfft=1200)
    mfccs = psf.mfcc(sig, samplerate=fs, numcep=13, nfft=1200, appendEnergy=False) # [929,13]
    
    fig, ax = plt.subplots()
    mfccs_data = np.swapaxes(mfccs, 0 ,1) # [13,929]
    # cax = ax.imshow(mfccs_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    # ax.set_title('MFCC')

    # plt.show()
    # plt.plot(mfccs_data)
    
    return mfccs_data



X=[]
Y=[]
BGN_Pub_copy=BGN_Pub.copy()


#start mixing, each person mix 10 sentence 
for i in range(1, 73):
    for j in range(1, 11):        
        wavfile.write(str(i)+"_"+str(j)+"_BGN.wav", BGN_Pub_copy[0], BGN_Pub_copy[1])
        x_mix, y_mix=create_training_example(BGN_Pub_copy, dic_person_label1[i], dic_person_label0[i], i, j)
        X.append(x_mix)
        Y.append(y_mix)
        
        BGN_Pub_tmp = wavfile.read("./data/Pub_10s.wav")
        BGN_Pub = np.copy(BGN_Pub_tmp)
        tmp3 = np.copy(np.array(BGN_Pub_tmp[1], dtype=np.float32))
        tmp4 = np.round(32767*tmp3)
        BGN_Pub[1] = tmp4.astype(np.int16)
        BGN_Pub[1] = BGN_Pub[1]
        BGN_Pub_copy=BGN_Pub.copy()
    

X=np.array(X)
Y=np.array(Y)
# X = X.astype(np.float16)
# Y = Y.astype(np.float16)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)


# np.savetxt('X_train.csv', X_train, delimiter=",")
# np.savetxt('X_test.csv', X_test, delimiter=",")

# np.savetxt('y_train.csv', y_train, delimiter=",")
# np.savetxt('y_test.csv', y_test, delimiter=",")

print(X.shape)
print(Y.shape)