# TSR-CNN
Traffic Sign Recognition Convolutional Neural Network
## Process
```
from google.colab import drive  
from google.colab import files  
```
```
from google.colab import drive   
drive.mount('/content/drive')   
```
```
!pip install -q kaggle   
uploaded = files.upload()   
```
```
!cp /content/kaggle.json ~/.kaggle/kaggle.json
```
```
!kaggle datasets download -d tamirpuzanov/road-signs-classification
```
```
%ls 
!unzip \*.zip && rm *.zip
```
```
!nvidia-smi
```
```python
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
import cv2
import numpy as np
from PIL import Image
import os 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
```

```python
train_path = '/content/drive/MyDrive/datasets/dataset/train/'
test_path = '/content/drive/MyDrive/datasets/dataset/test/'
```
```python
img = load_img(train_path + "Autocesta/037b8446-0386-48f4-a3a1-05e4d8ea27f9.png", target_size=(100,100))
plt.imshow(img)
plt.axis("off")
plt.show()
#Printing the shape of the image array 
x = img_to_array(img)
print(x.shape)
```
```python
batch_size = 50

train_generator = ImageDataGenerator().flow_from_directory(
directory = train_path,
target_size= x.shape[:2],
batch_size = batch_size,
color_mode= "rgb",
class_mode= "categorical")

test_generator = ImageDataGenerator().flow_from_directory(
directory = test_path,
target_size= x.shape[:2],
batch_size = batch_size,
color_mode= "rgb",
class_mode= "categorical")
```
```python
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(136, activation='softmax'))
```
```python
model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
```
```python
epochs = 50
```
```python
with tf.device('/gpu:0'):
  model.fit(
  x = train_generator,
  steps_per_epoch = 1600 // batch_size,
  epochs=epochs,
  validation_data = test_generator,
  validation_steps = 800 // batch_size)
```
```python  
model.save('drive/MyDrive/datasets/model_fuzzy.h5')
```
```python
model.save('drive/MyDrive/datasets/model_fuzzy2.h5')
```
```python
model.save('drive/MyDrive/datasets/model_TPU.h5')
```
```python
model.save('drive/MyDrive/datasets/model_GTX960.h5')
```
```python  
model1 = load_model('drive/MyDrive/datasets/model_fuzzy.h5')
model2 = load_model('drive/MyDrive/datasets/model_fuzzy2.h5')
```
```python
test_img = load_img("/content/drive/MyDrive/datasets/right-turn-prohibited-road-sign-uk-2CBGK16.jpg", target_size=(100,100))
test_img_array = img_to_array(test_img)
test_img_expanded = np.expand_dims(test_img_array, axis = 0)
```
```python
prediction = model2.predict(test_img_expanded)
prediction = prediction.tolist()
prediction = prediction[0]
index = prediction.index(max(prediction))
print(index)
```
```python
labels = ["Autocesta", "Benzinska postaja", "Biciklisti na cesti", "Biciklistička staza", "Bolnica", "Brza cesta", "Brzina koja se preporucuje-40", "Brzina koja se preporucuje-50", 
          "Brzina koja se preporucuje-60", "Brzina koja se preporucuje-70", "Brzina koja se preporucuje-80", "Cesta s prednošću prolaska", "Cesta s jednosmjernim prometom 1",
          "Cesta s jednosmjernim prometom 2", "Cesta s jednosmjernim prometom 3", "Divljac na cesti", "Djeca na cesti", "Domace zivotinje na cesti", "Dopusteni smjerovi lijevo i desno",
          "Dopusteni smjerovi naprijed i desno", "Dopusteni smjerovi naprijed i lijevo", "Dopusteno obilazenje", "Dvostruki zavoj ili vise uzastopnih zavoja od kojih je prvi udesno",
          "Dvostruki zavoj ili vise uzastopnih zavoja od kojih je prvi uljievo", "Hotel ili motel", "Izbocina na cesti", "Izmjenicno parkiranje 1", "Izmjenicno parkiranje 2",
          "Kamenje pada", "Kamenje prsti", "Kontrolna tocka za medunarodni cestovni prijevoz", "Kruzni tok prometa", "Nailazak na prometna svjetla", "Najmanja udaljenost izmedu vozila",
          "Neravan kolnik 1", "Neravan kolnik 2", "Obavezno zaustavljanje", "Obiljezen pjesacki prelaz 1", "Obiljezen pjesacki prelaz 2", "Obvezan smjer - desno 1", "Obvezan smjer - desno 2",
          "Obvezan smjer - lijevo 1", "Obvezan smjer - lijevo 2", "Obvezan smjer - naprijed", "Obvezno obilazenje s desne strane", "Obvezno obilazenje s lijeve strane", "obvezno zaustavljanje",
          "Ogranicenje brzine-110", "Ogranicenje brzine-40", "Ogranicenje brzine-50", "Ogranicenje brzine-60", "Ogranicenje brzine-70", "Ogranicenje brzine-80", "Ogranicenje brzine-90", 
          "Opasna nizbrdica", "Opasna uzbrdica", "Opasnost na cesti", "Otvaranje prometne trake", "Parkiraliste 1", "Parkisraliste 2", "Pjesacka i biciklisticka staza 1", "Pjesacka i biciklisticka staza 2",
          "Pjesacka i biciklisticka staza 3", "Pjesacka staza", "Pjesacka zona", "Podrucje smirenog prometa", "Podzemni ili nadzemni pjesacki prelaz 1", "Podzemni ili nadzemni pjesacki prelaz 2",
          "Policija", "Praonica vozila", "Prednost prolaska prema vozilima iz suprotnog smjera", "Prednost prolaska za vozila iz suprotnog smjera", "Prestanak ogranicenja brzine-40",
          "Prestanak ogranicenja brzine-50", "Prestanak ogranicenja brzine-60", "Prestanak ogranicenja brzine-70", "Prestanak svih zabrana", "Prestanak zabrane pretjecanja svih motornih vozila",
          "Prijelaz ceste preko zeljeznicke pruge s branikom", "Promet u oba smjera", "Putna patrolna postaja", "Radionica za oporavak vozila", "Radovi na cesti", "Raskrizje s cestom koja ima prednost prolaza",
          "Raskrizje s kruznim tokom prometa", "Raskrizje sa sporednom cestom pod pravim kutom", "Restoran", "Sklizak kolnik", "Slijepa cesta 1", "Slijepa cesta 2", "Slijepa cesta 3",
          "Spajanje sporedne ceste pod pravim kutom", "Spajanje sporedne ceste sa lijeve strane", "Stajaliste autobusa", "Stajaliste tramvaja", "Suzenje ceste", "Suzenje ceste s desne strane",
          "Suzenje ceste s lijeve strane", "Taxi", "Telefon", "Teren ureden za izletnike", "Tramvajska pruga", "Tunel", "Ustanoca hitne medicinske pomoci", "WC", "Zabrana parkiranja", "Zabrana polukruznog okratanja",
          "Zabrana pretjecanja svih motornih vozila bez prikolice i mopeda", "Zabrana prolaska bez zaustavljanja", "Zabrana prometa u jednom smjeru", "Zabrana prometa u oba smjera",
          "Zabrana prometa za bicikle", "Zabrana prometa za mopede", "Zabrana prometa za pjesake", "Zabrana prometa za sva motorna vozila koja vuku prikljucno vozilo", "Zabrana prometa za teretne automobile",
          "Zabrana prometa za vozila koja prevoze eksploziv ili neke zapaljive tvari", "Zabrana prometa za vozila koja prevoze opasne tvari", "Zabrana skretanja udesno", "Zabrana skretanja ulijevo",
          "Zabrana zaustavljanja i parkiranja", "Zatvaranje prometne trake", "Zavoj udesno", "Zavoj ulijevo", "Zavrsetak autoceste", "Zavrsetak biciklisticke staze", "Zavrsetak brze ceste",
          "Zavrsetak ceste s prednoscu prolaska", "Zavrsetak ceste s jednosmjernim prometom", "Zavrsetak pjesacke i biciklisticke staze", "Zavrsetak podrucja smirenog prometa",
          "Zeljeznicka pruga", "Zona u kojoj je ogranicena brzine-20", "Zona u kojoj je ogranicena brzina-30", "Zona u kojoj je ogranicena brzina-40", "Zona u kojoj je ogranicena brzina-50"]
```
```python          
img = load_img("/content/drive/MyDrive/datasets/right-turn-prohibited-road-sign-uk-2CBGK16.jpg", target_size=(100,100))
plt.imshow(img)
plt.axis("off")
plt.show()
#Printing the shape of the image array 
x = img_to_array(img)
print(x.shape)
```
```python
predicted_image = labels[index]
print(predicted_image)
```
```python
model2.summary()
```
```python
print(tf.__version__)
```
