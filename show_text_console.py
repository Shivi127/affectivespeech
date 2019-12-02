import time
import sys
from sound_state import *

import logging
logging.getLogger().setLevel(logging.INFO)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def draw_text_on_console(lines, state=STATE_VOLUME_CONSTANT):
    if state is STATE_VOLUME_ELEVATED:
        prefix = color.BOLD
        suffix = color.END
    elif state is STATE_VOLUME_LOWERED:
        prefix = color.UNDERLINE
        suffix = color.END
    else:
        prefix = ''
        suffix = ''
    print('\033[H\033[J')
    for y in range(len(lines)):
        print(prefix+lines[y]+suffix)
	
if len(sys.argv) > 1:
    draw_text_on_console(sys.argv[1:])
