import os
import sys
import numpy as np

from mido import MidiFile, MidiTrack, Message

working_directory = './dataset'

## sample domensions
number_of_notes = 96

sample_per_measure = 96



def midi_to_samples(filename):
	mid = MidiFile(filename)






if __name__ == '__main__': 
    paths = []
    for root, dirs, files in os.walk(working_directory):
    	for file in files:
    		if file.endswith('.mid'):
    			paths.append(os.path.join(root, file))

    paths.sort()
    print('Found', len(paths), 'midi files in', working_directory)
    print('\n')

    log = open('tsign.txt', 'w')

    for k, filename in enumerate(paths):
    	try:
    		mid = MidiFile(filename)
    	except (IOError, KeyError, ValueError) as e:
    		log.write('error')
    		continue

    	print('Processing', str(k), '/', str(len(paths)))
    	sys.stdout.write("\033[F")
    	
    	ticks_per_beat = mid.ticks_per_beat

    	tsign = [
    		(msg.numerator, msg.denominator) 
    		for track in mid.tracks 
    		for msg in track if msg.is_meta and msg.type == 'time_signature'
    	]

    	log.write(str(tsign)+', '+str(ticks_per_beat)+'\n')

    log.close()

    	

    