import os
import sys
import numpy as np

from mido import MidiFile, MidiTrack, Message
from scipy.misc import imsave

working_directory = './dataset'
saving_directory = './raw'

## starting sample index
offset = 0

## sample domensions
samples_per_measure = 96

## number of noted that count
nb_note = 96


## Raised when multiple time signature is detected
class MultiTimeSignature(Exception):
	pass   

## Convert MIDI files to samples
def midi_to_samples(filename, info=None):
	mid = MidiFile(filename)

	sign = [
		(msg.numerator, msg.denominator) 
		for track in mid.tracks 
		for msg in track 
		if msg.is_meta and msg.type == 'time_signature'
	]

	if not sign: 
		seg = (4, 4)
	elif len(sign)==1:
		seg = sign[0]
	else:
		for seg in sign:
			if seg!=sign[0]: seg = None; break 

	if not seg: raise MultiTimeSignature

	ticks_per_sample = int(mid.ticks_per_beat * seg[0] / samples_per_measure)
	if not ticks_per_sample: raise ZeroDivisionError

	Seq = [[ ] for j in range(nb_note)]
	max_k = 0
	for track in mid.tracks:
		tick = 0
		for msg in track:
			tick += msg.time
			if msg.is_meta: continue 
			elif msg.type=='note_on' and msg.velocity:
				k = tick//ticks_per_sample
				n = msg.note - 21
				if k not in Seq[n]: Seq[n].append(k)
				if k > max_k: max_k = k

	d = max_k // samples_per_measure + 1
	samples = np.zeros((nb_note, d * samples_per_measure), dtype=np.uint8)

	for k, seq in enumerate(Seq):
		if not seq: continue
		for idx in seq:
			samples[k, idx] = 1

	samples = np.hsplit(samples, d)

	directory = filename.split('/')[-2]
	if not os.path.exists(os.path.join(saving_directory, directory)):
		os.makedirs(os.path.join(saving_directory, directory))

	np.save(os.path.join(saving_directory, directory, filename.split('/')[-1]), samples)


# log = open('tsign.txt', 'w')

# for k, filename in enumerate(paths):
# 	try:
# 		mid = MidiFile(filename)
# 	except (IOError, KeyError, ValueError, IndexError, EOFError) as e:
# 		log.write('error: '+str(e)+'\n')
# 		continue

# 	print('Processing', str(k), '/', str(len(paths)))
# 	sys.stdout.write("\033[F")

# 	ticks_per_beat = mid.ticks_per_beat

# 	tsign = [
# 		(msg.numerator, msg.denominator) 
# 		for track in mid.tracks 
# 		for msg in track if msg.is_meta and msg.type == 'time_signature'
# 	]

# 	log.write(str(tsign)+', '+str(ticks_per_beat)+'\n')

# log.close()

if __name__ == '__main__': 
	paths = []
	for root, dirs, files in os.walk(working_directory):
		for file in files:
			if file.endswith('.mid'):
				paths.append(os.path.join(root, file))

	paths.sort()
	print('Found', len(paths), 'midi files in', working_directory)

	log = open('log.txt', 'a')
	for k, filename in enumerate(paths[offset:]):
		try:
			midi_to_samples(filename)
		except(IOError, KeyError, ValueError, IndexError, EOFError, ZeroDivisionError) as e:
			log.write(filename+', error type: '+str(e)+'. \n')
			continue
		except MultiTimeSignature:
			log.write(filename+', multiple time signature detected. \n')
			continue

		# print(len(s))
		# for i in range(len(s)):
		# 	# print(s[i].shape)
		# 	savename = filename.split('/')[-1]+'_'+str(i)+'.png'
		# 	imsave(savename, 255-s[i])

		print('Processing', str(offset+k+1), '/', str(len(paths)))
		sys.stdout.write("\033[F")


	log.close()
	print('\nAll Done.')



