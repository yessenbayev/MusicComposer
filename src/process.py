import os
import sys
import numpy as np
import pandas as pd

import midi

from mido import MidiFile, MidiTrack, Message, tempo2bpm
from scipy.misc import imsave

working_directory = './dataset'
saving_directory = './raw'


## sample domensions
samples_per_measure = 96

## number of noted that count
numver_of_nodes = 96


## Raised when time signature issue occurs
class TimeSignatureError(Exception):
	pass   

## Raised when tempo issue occurs
class TempoError(Exception):
	pass

## Rasied when asynchronous is detcetd
class AsynchronousTracks(Exception):
	pass


## main entry of functions
if __name__ == '__main__': 
	# paths = []
	# for root, dirs, files in os.walk(working_directory):
	# 	for file in files:
	# 		if file.endswith('.mid'):
	# 			paths.append(os.path.join(root, file))

	# paths.sort()
	# print('Found', len(paths), 'midi files in', working_directory)
	# print('\n')

	# log = open('log.txt', 'w')
	# for k, filename in enumerate(paths[offset:]):
	# 	try:
	# 		midi_to_samples(filename)
	# 	except(IOError, KeyError, ValueError, IndexError, EOFError, ZeroDivisionError) as e:
	# 		log.write(filename+', '+str(e)+'. \n')
	# 		continue
	# 	except TimeSignatureError:
	# 		log.write(filename+', multiple time signature detected. \n')
	# 		continue
	# 	except TempoError:
	# 		log.write(filename+', multiple tempo signature detected. \n')
	# 		continue
	# 	except AsynchronousTracks:
	# 		log.write(filename+', asynchronous detected. \n')
	# 		continue			

	# 	# for i in range(len(s)):
	# 	# 	# print(s[i].shape)
	# 	# 	savename = filename.split('/')[-1]+'_'+str(i)+'.png'
	# 	# 	imsave(savename, 255-s[i])

	# 	print('Processing', str(offset+k+1), '/', str(len(paths)))
	# 	sys.stdout.write("\033[F")


	# log.close()
	# print('\n\nAll Done.')

	datainfo = pd.read_excel('midi_info.xlsx')
	datainfo = datainfo.drop(columns=['Unnamed: 0'])

	paths = [
		item[0] for item in np.array(datainfo) if item[-1]==1
	]

	for k, filename in enumerate(paths):
		try:
			samples = midi.midi_to_samples(filename)
		except(IOError, KeyError, ValueError, IndexError, EOFError, ZeroDivisionError) as e:
			continue
		except TimeSignatureError:
			continue
		except TempoError:
			continue
		except AsynchronousTracks:
			continue	

		directory = filename.split('/')[-2]
		if not os.path.exists(os.path.join(saving_directory, directory)):
			os.makedirs(os.path.join(saving_directory, directory))

		np.save(os.path.join(saving_directory, directory, filename.split('/')[-1]), samples)

		print('Processing', str(k+1), '/', str(len(paths)))
		sys.stdout.write("\033[F")

	print('All Done. ')




