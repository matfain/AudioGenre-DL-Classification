
import pandas as pd
import numpy as np
import librosa
import librosa.display
import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset

#defining custom dataset to input to model
class MusicGenres(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
    self.genre_dict = {"blues":0,"classical":1,"country":2,"disco":3,"hiphop":4,"jazz":5,"metal":6,"pop":7,"reggae":8,"rock":9}

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):
    name = str(self.annotations.iloc[index,0])
    y_label = self.genre_dict[str(self.annotations.iloc[index,59])]   # genre col converted to class label using genre dict
    im = torch.load(self.root_dir+name+'.pt') #loading saved torch tensor representing file
    im = torch.unsqueeze(im,axis=0)  #adding 3rd dimension
    im = torch.tile(im,(3,1,1)) #repeating 2 dimensional array across 3 dimensions to resemble image input for CNNs based models
    if self.transform:
      im = self.transform(im)

    return (im, y_label)
  

 #Model based on resnet with option to train all the model or use pretrained weights up untill the classification layer 

class GenreClassifier(nn.Module):
  def __init__(self, num_classes=10, train_CNN=False):
    super().__init__()
    resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')                    
    for param in resnet.parameters():
            param.requires_grad_(train_CNN)
    
    modules = list(resnet.children())[:-1]
    self.resnet = nn.Sequential(*modules)
    self.logits = nn.Linear(resnet.fc.in_features, num_classes)

  def forward(self, x):
     res_features = self.resnet(x)
     reshaped = res_features.view(res_features.size(0), -1)
     out = self.logits(reshaped)
     return out
  

# Calculatin training set mean and std for applying custom transformation later
# https://kozodoi.me/blog/20210308/compute-image-stats#3.-Computing-image-stats

if __name__ == "__main__":

  path_tensors = r"C:\Users\matan\Desktop\MLDL_Projects\Audio_Classification\Data\spectrogram_tensors\\"
  csv_file = r"C:\Users\matan\Desktop\MLDL_Projects\Audio_Classification\Data\features_3_sec.csv"

  resize_transform = transforms.Compose([transforms.Resize((224, 224))])    # Resizing for pre=trained model


  base_ds = MusicGenres(csv_file= csv_file, root_dir= path_tensors, transform= resize_transform)
  train_ds, test_ds = torch.utils.data.random_split(base_ds , [0.85, 0.15], generator=torch.Generator().manual_seed(42))

  train_loader = DataLoader(train_ds, batch_size= 64, shuffle= True)
  test_loader = DataLoader(test_ds, batch_size= 64, shuffle= True)

  sum = torch.zeros(3)
  sum_sq = torch.zeros(3)
  count = len(train_ds)*224*224

  for imgs, _ in train_loader:
    sum += imgs.sum(dim= [0,2,3])
    sum_sq += (imgs**2).sum(dim= [0,2,3])

  total_mean = sum/count
  total_var = (sum_sq/count) - (total_mean**2)
  total_std = torch.sqrt(total_var)

  print(f'mean is: {str(total_mean)}')    # tensor([-39.0525, -39.0525, -39.0525])
  print()
  print(f'std is: {str(total_std)}')      # tensor([14.9446, 14.9446, 14.9446])