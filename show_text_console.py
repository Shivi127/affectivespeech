import time
import sys
import logging
logging.getLogger().setLevel(logging.INFO)

def draw_text(lines):
	print('\033[H\033[J')
	for y in range(len(lines)):
		print(lines[y])
	
if len(sys.argv) > 1:
	draw_text(sys.argv[1:])
