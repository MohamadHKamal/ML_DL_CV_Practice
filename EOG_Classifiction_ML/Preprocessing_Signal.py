from scipy.signal import butter, lfilter
import numpy as np
import  pywt
class Preprocessing():

    def __init__(self):
        return

    def Normalize(self,signal,signal_len):

        min_val = np.min(signal)
        max_val = np.max(signal)
        for i in range(signal_len):
            signal[i] = (signal[i]-min_val) / (max_val - min_val)

        return signal

    def Mean_Removal(self,signal,signal_len):

        signal_mean=0

        for i in range(signal_len):
            signal_mean = signal_mean + signal[i]

        signal_mean = signal_mean / signal_len

        for i in range(signal_len):
            signal[i] = signal[i] - signal_mean

        return signal

    def DowmSampling(self,signal,signal_len,DowmSampling_amount):

        down_sampled_signal=[]

        for i in range(0,signal_len,DowmSampling_amount):
            down_sampled_signal.append(signal[i])


        return np.array(down_sampled_signal)

    def BandPass(self,signal,lowcut, highcut,sampling_rate,order=1):

        nyq =  sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band') # return the low and high pass filter

        filterd_signal = lfilter(b, a, signal)

        return filterd_signal
    def feature_Extraction(self,signal,levels,wavelet_name):

        list_An_Di=[]
        features=[]
        list_An_Di=pywt.wavedec(data=signal,wavelet=wavelet_name,level=levels)
        for arr in list_An_Di:
            features = features + list(arr)

        return np.array(features)