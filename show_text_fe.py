import sys
import subprocess
from show_text_console import draw_text_on_console

from sound_state import *

LINES, LINE_LENGTH = subprocess.check_output(['stty', 'size']).split()
LINES=int(LINES)
LINE_LENGTH=int(LINE_LENGTH)
LINES -= 1
print("lines: {} cols: {}".format(LINES,LINE_LENGTH))
DISPLAY_COMMAND = draw_text_on_console

PAD_LINES = ['' for i in range(LINES)]
def show_text(text, state=STATE_VOLUME_CONSTANT):
  words = text.split(' ')
  tokens = []
  for word in words:
    tokens += list(chunkstring(word, LINE_LENGTH))

  lines = []
  line = ''
  for word in tokens:
    if len(word) + len(line) >= LINE_LENGTH:
      lines.append(line.strip())
      line = ''
    if len(line) == 0:
      line = word
    else:
      line = line + ' ' + word
  if len(line) > 0:
    lines.append(line.strip())
    
  print_lines(lines, state)

def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def print_lines(text, state):
  lines = PAD_LINES + text
  lines = lines[-LINES:] 
  draw_text_on_console(lines, state)

if __name__  == "__main__":
	if len(sys.argv) > 1:
		show_text(" ".join(sys.argv[1:]))
	else:
		show_text('123456789012345 that is some long text')
