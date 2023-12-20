### AudioGenre DL Classification Project
In this project I used transfer learning via Resnet18 CNN for a music genre classification task.  
I used the GTZAN dataset so if you are interested in doing a simillar project you can find the relevant data here:  
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

#### Project overview & workflow:
* Data preprocessing - spectographs, corrupted files debugging, Dataset&Dataloader configuration etc
* Model selection and training - trying 2 different approaches as shown in the main notebook
* Final Analysis - summary using confusion matrix & forward suggestions

  In the figures below you see the final confusion matrix along with an inferenced test sample & TB graphs from the training.

<p align="center">
<img src="https://github.com/matfain/AudioGenre-DL-Classification/assets/132890076/425f4ba0-2f46-4e19-8426-517c8d6be948" width="500" height="500">   <img src="https://github.com/matfain/AudioGenre-DL-Classification/assets/132890076/4284bf9d-683b-41f0-b66f-761f42c8b994" width="300" height="400">
<img src="https://github.com/matfain/AudioGenre-DL-Classification/assets/132890076/4e2364e9-63dd-46f7-b639-ad90a2b8fbc7" width="352" height="242">   <img src="https://github.com/matfain/AudioGenre-DL-Classification/assets/132890076/1a5eee22-f757-4378-9646-8e65f5935c5e" width="352" height="242">
<p/>
Additional credit goes to https://github.com/natcasd/DL-Final-Project/tree/main
