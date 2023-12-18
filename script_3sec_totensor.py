#Script credit goes to https://devpost.com/software/music-genre-classification-dl-final-project#updates
import pandas as pd
import librosa
import numpy as np
import torch
import os

path = r"C:\Users\matan\Desktop\MLDL_Projects\Audio_Classification\Data\genres_original\\"  #path to .wav files
csv_file = r"C:\Users\matan\Desktop\MLDL_Projects\Audio_Classification\Data\features_3_sec.csv" # path to csv relevant file
pathspectrograms = r"C:\Users\matan\Desktop\MLDL_Projects\Audio_Classification\Data\spectrogram_tensors\\"
files = pd.read_csv(csv_file) #read .csv file of labels and names of 3 second audio files

numrows = len(files.index)
hop_length = 512
n_fft = 2048


#.wav files are 30 seconds long, filenames follow the format "genre.number.wav", have to split these files into 3 second mel spectrograms with filename format "genre.number.number_in_split.wav"
#the filenames for the 3 second mel spectrograms are given in the .csv file, but have to pull the data for each of them from their corresponding 30 second .wav file

for index in range(numrows): #iterate through every file in .csv
  if not os.path.exists(pathspectrograms+str(files.iloc[index,0])+'.pt'):  #check if file has already been generated
    print(str(files.iloc[index,0]))
    name = str(files.iloc[index,0]).split('.')
    numberinfile = float(name[2]) #get number_in_split from file
    snipduration = 3. #spectrograms being generated will be 3 seconds long
    file = name[0] + '.' + name[1] + '.' + name[3] #concatenate 30 second file name from the 3 second file name
    img_path = path + str(files.iloc[index,59]) + "\\" + file #get file path of 30 second wav file
    y,s = librosa.load(img_path, offset = numberinfile*3, duration = snipduration) #load selected 3 second snippet of .wav
    mel = librosa.feature.melspectrogram(y=y, sr=s, hop_length=hop_length, n_fft=n_fft, n_mels = 96)
    spectrogram = np.abs(mel) 
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max) #convert to mel spectrogram
    tensor = torch.from_numpy(power_to_db)
    torch.save(tensor, pathspectrograms+str(files.iloc[index,0])+'.pt') #save as torch tensor