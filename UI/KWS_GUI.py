import torch
from torch import nn

import wave
import pyaudio
import time
import winsound

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import python_speech_features as psf

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def audio_record():
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "voice.wav"

    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    idx = p.get_default_input_device_info()["index"]

    for i in range(device_count):
        print("Input device id", i, "--", p.get_device_info_by_index(i)["name"])
    print("Default device index:", idx)

    ans = input("Do you want to change input device?(Y/N)\n")
    if ans == "Y":
        idx = input("Which device do you prefer for recording(type index)?")
    
    print(p.get_device_info_by_index(int(idx)))
    RATE = int(p.get_device_info_by_index(int(idx))["defaultSampleRate"])
    
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, input_device_index=int(idx),
                    frames_per_buffer=CHUNK)

    Recordframes = []

    winsound.PlaySound('alert',winsound.SND_ASYNC)
    time.sleep(0.5)
    print("Start recording....")
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK,exception_on_overflow = False)
        Recordframes.append(data)

    print("Recording finsihed....")
    stream.stop_stream()
    stream.close()

    p.terminate()

    wavfile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(p.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(Recordframes))
    wavfile.close()

def MFCC_feature_extraction_psf(wav_file):
    fs = 48000
    fs, sig = wavfile.read(wav_file)
    
    #print("fs:", fs)
    #print(sig.shape)
    sig = sig[:,0]+sig[:,1]
    #print(sig.shape)
    
    mfccs = psf.mfcc(sig, samplerate=fs, numcep=13, nfft=1200, appendEnergy=False) # [999,13]
    mfccs_data = np.swapaxes(mfccs, 0 ,1) # [13,999]
    
    return mfccs_data


def detect_triggerword(wav_file, model, threshold):
    x = MFCC_feature_extraction_psf(wav_file)
    
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = np.expand_dims(x, axis=0)
    
    predictions = model(torch.tensor(x).float())
    predictions = predictions.detach().numpy()
    
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    
    predictions[predictions > threshold] = 1
    if 1 in predictions[0,:,0]:
        return True
    
    return False

class Delta_RNN(nn.Module):
  def __init__(self):
    super(Delta_RNN, self).__init__()
    
    self.loss_fct = nn.BCELoss()
    self.conv = nn.Sequential(
        nn.Conv1d(in_channels=13, out_channels=26, kernel_size=15, stride=4),
        nn.BatchNorm1d(num_features=26), #(batch size,num_channel,length)
        nn.ReLU(),
        nn.Dropout(p=0.8),
    )
    
    self.biRNN = nn.GRU(input_size=26,hidden_size=13,batch_first=True,dropout=0.2,bidirectional=True)
    
    self.dropout = nn.Dropout(p=0.8)
    self.batchNorm1 = nn.BatchNorm1d(num_features=26)
    self.batchNorm2 = nn.BatchNorm1d(num_features=16)
    
    self.RNN = nn.GRU(input_size=26,hidden_size=16,batch_first=True)
    
    self.classifier = nn.Sequential(
        nn.Linear(in_features=16,out_features=1),
        nn.Sigmoid(),
    )

  def forward(self, x, labels=None):
    x = self.conv(x)
    
    # [batch size, num channel, sequence length] to 
    # [batch size, sequence length, input size]
    x = torch.swapaxes(x,1,2)
    residual_x, hn = self.biRNN(x)
    
    x += residual_x.detach()
    x, hn = self.biRNN(x)
    x = torch.swapaxes(x,1,2)
    
    x = self.dropout(x)
    x = self.batchNorm1(x)
    
    x = torch.swapaxes(x,1,2)
    x,hn = self.RNN(x)
    x = torch.swapaxes(x,1,2)
    
    x = self.dropout(x)
    x = self.batchNorm2(x)
    x = self.dropout(x)
    
    x = torch.swapaxes(x,1,2)
    
    outputs = self.classifier(x)
    
    if labels != None:
      loss = self.loss_fct(outputs, labels)

      return outputs, loss
    
    return outputs # [number of data, 247, 1]
    
class Beta_RNN(nn.Module):
    def __init__(self):
        super(Beta_RNN, self).__init__()

        self.loss_fct = nn.BCELoss()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=13, out_channels=26, kernel_size=15, stride=4),
            nn.BatchNorm1d(num_features=26), #(batch size,num_channel,length)
            nn.ReLU(),
            nn.Dropout(p=0.8),
        )

        self.RNN1 = nn.GRU(input_size=26,hidden_size=16,batch_first=True)

        self.dropout = nn.Dropout(p=0.8)
        self.batchNorm = nn.BatchNorm1d(num_features=16)

        self.RNN2 = nn.GRU(input_size=16,hidden_size=16,batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16,out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels=None):
        x = self.conv(x)

        # [batch size, num channel, sequence length] to 
        # [batch size, sequence length, input size]
        x = torch.swapaxes(x,1,2)
        x,hn = self.RNN1(x)

        x = torch.swapaxes(x,1,2)

        x = self.dropout(x)
        x = self.batchNorm(x)

        x = torch.swapaxes(x,1,2)
        x,hn = self.RNN2(x)

        x = torch.swapaxes(x,1,2)
        x = self.dropout(x)
        x = self.batchNorm(x)
        x = self.dropout(x)

        x = torch.swapaxes(x,1,2)

        outputs = self.classifier(x)

        if labels != None:
            loss = self.loss_fct(outputs, labels)

            return outputs, loss

        return outputs # [number of data, 247, 1]



if __name__ == "__main__":
    device = "cpu"
    
    #net = Beta_RNN()
    #checkpoint = torch.load('GRU_MFCC_normalized(RMS)_BGN-7db_0.6sec_HalfLabel0-insert[0,4]_low-FA.pt',  map_location=torch.device('cpu'))
    
    net = Delta_RNN()
    checkpoint = torch.load('delta-GRU_MFCC_normalized(RMS)_BGN-7db_0.6sec_HalfLabel0-insert[0,4]_low-FA.pt',  map_location=torch.device('cpu'))
    
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()
    #print(net)
    
    # Set probability threshold in GUI
    threshold = 0.4
    select = int(input("You want the system to be sensitive or accurate? (1:sensitive, 2:accurate)\n"))
    if select == 1:
        threshold = 0.2
    elif select == 2:
        threshold = 0.4
    
    #threshold = input("Choose your threshold for positive detection, 0.1-0.5 recommend?\n")
    
    print("Threshold:", threshold)
    # Take in microphone input
    audio_record()
    
    # Run model
    res = detect_triggerword("voice.wav", net, float(threshold))
    
    if res:
        print("Activate！")
    else:
        print("Not activate!")
    