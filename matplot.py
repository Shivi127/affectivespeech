import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import time

def draw_bar(values):
  intervals = range(-1*len(values) + 1, 1)
  y_pos = np.arange(len(intervals))

  plt.bar(y_pos, values, align='center', alpha=0.5)
  plt.xticks(y_pos, intervals)
  plt.ylabel('interval')
  plt.title('value')

  print values
  plt.draw()

values = [10,8,1,4,2,6]
for step in range(10):
  plt.cla()
  draw_bar(values)
  values = values[1:]+[step]
  plt.pause(.2)
plt.show()
