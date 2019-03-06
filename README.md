# ECE 271B Final Project
## Downloading the dataset
run </br>
conda install requests </br>
conda install -c anaconda beautifulsoup4 </br>
conda install html5lib </br>
then run </br>
python download_dataset.py </br>

## Installing the required packages

### mido
conda install -c roebel mido </br>


## DataGenerator Method

### import
from DataGenerator import MidiDataGenerator
### initilize with the root path and the length of desired measures $m$
midi = MidiDataGenerator('./raw', m=16)
### generate a tensor of shape (10, m, 96, 96)
sample = midi.samples(size=10)


## Training

### run
python SeqTrain.py tasks.txt

### tasks.txt contains the path of dataset

## Generating Random Songs

see test.ipynb
