import numpy as np
from mido import MidiFile, MidiTrack, Message


## the desired output feature shape
number_of_notes = 96
samples_per_measure = 96

## Convert MIDI files to samples
def midi_to_samples(filename, info=None):
	mid = MidiFile(filename)
	if mid.type==2: raise AsynchronousTracks

	signature = [
		(msg.numerator, msg.denominator) 
		for track in mid.tracks 
		for msg in track 
		if msg.is_meta and msg.type == 'time_signature'
	]

	if not signature: seg = (4, 4)
	else:
		seg = signature[0]
		for k in range(1, len(signature)):
			if seg!=signature[k]: 
				seg = None
				break 

	if not seg: raise TimeSignatureError

	ticks_per_sample = int(mid.ticks_per_beat * seg[0] / samples_per_measure)
	if not ticks_per_sample: raise ZeroDivisionError

	Seq = [[ ] for j in range(number_of_notes)]
	max_k = 0
	
	for track in mid.tracks:
		tick = 0
		for msg in track:
			tick += msg.time
			# if msg.is_meta: continue 
			if msg.type=='note_on' and msg.velocity:
				k = tick//ticks_per_sample
				n = msg.note - 21
				if k not in Seq[n]: Seq[n].append(k)
				if k > max_k: max_k = k

	d = max_k // samples_per_measure + 1

	samples = np.zeros((number_of_notes, d * samples_per_measure), dtype=np.uint8)

	for k, seq in enumerate(Seq):
		if not seq: continue
		for idx in seq:
			samples[k, idx] = 1

	samples = np.vsplit(samples.T, d)

	return samples


def samples_to_midi(samples, filename, thresh=0.5):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)

	ticks_per_sample = 5 * mid.ticks_per_beat / samples_per_measure

	last_ticks, ticks = 0, 0
	for k, sample in enumerate(samples):
		for y in range(sample.shape[0]):
			ticks += ticks_per_sample
			for x in range(sample.shape[1]):
				note = x + 21
				# determine if the node on
				if sample[y, x] >= thresh and (y==0 or sample[y-1,x] < thresh):
					track.append(Message('note_on',  note=note, velocity=127, time=int(ticks-last_ticks)))
					last_ticks = ticks
				if sample[y, x] >= thresh and (y==sample.shape[0]-1 or sample[y+1,x] < thresh):
					track.append(Message('note_off', note=note, velocity=127, time=int(ticks-last_ticks)))
					last_ticks = ticks


	mid.save(filename)


def midi_to_dataframe(filename, info=None):
	## filename, time_signature, bpm, length (in sec), num of measures, is_synchronous
	mid = MidiFile(filename)
	if mid.type==2: raise AsynchronousTracks

	signature = [
		(msg.numerator, msg.denominator) 
		for track in mid.tracks 
		for msg in track 
		if msg.is_meta and msg.type == 'time_signature'
	]

	tempo = [
		msg.tempo 
		for track in mid.tracks 
		for msg in track 
		if msg.is_meta and msg.type == 'set_tempo'
	]

	if not signature: seg = (4, 4)
	else:
		seg = signature[0]
		for k in range(1, len(signature)):
			if signature[k]!=seg:
				seg = None
				break

	if not tempo: tpo = None
	else:
		tpo = tempo[0]
		for k in range(1, len(tempo)):
			if tempo[k]!=tpo:
				tpo = None
				break

	if not seg: raise TimeSignatureError
	if not tpo: raise TempoError

	bpm = tempo2bpm(tpo)
	length = mid.length

	return str(seg), bpm, length, np.ceil(bpm * length / 60 / seg[0]), mid.type
