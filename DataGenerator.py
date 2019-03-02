from os import walk, path
from collections import deque
import numpy as np


class MidiDataGenerator():
	def __init__(self, root_dir, m=16):
		self.paths = []
		for root, _, files in walk(root_dir):
			for file in files:
				if file.endswith('.npy'):
					self.paths.append(path.join(root, file))

		self.queue = deque()
		self.pos = 0
		self.len = m

	def format_data(self, data):
		m_i = data.shape[0]
		if m_i > self.len:
			tmpArr = np.vstack([
				data, 
				np.zeros((self.len - m_i % self.len,96,96), dtype=np.uint8)
			]) if m_i % self.len else data
			return np.vsplit(tmpArr, tmpArr.shape[0] // self.len)
		
		# in case when m_i < m
		return [np.vstack([
			data for k in range(self.len // m_i)
		])] if not self.len % m_i \
		else [np.vstack([
			np.vstack([data for k in range(self.len // m_i)]),
			np.zeros((self.len % m_i,96,96), dtype=np.uint8)
		])]

	def samples(self, size=10):
		while len(self.queue) < size:
			data = np.load(self.paths[self.pos])
			self.pos = (self.pos + 1) % len(self.paths)
			for sample in self.format_data(data):
				self.queue.append(sample)

		return np.array([
			self.queue.popleft() for k in range(size)
		]), self.pos











