import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import time

def draw_graph(title, y_limits, y_label, bar_values, line_values, x_labels=None):
  intervals = range(-1*len(bar_values) + 1, 1)
  plt.cla()
  y_pos = np.arange(len(intervals))
  if x_labels is None:
    x_labels = intervals
  plt.xticks(y_pos, x_labels)
  plt.ylabel(y_label)
  plt.title(title)

  plt.bar(y_pos, bar_values, align='center', alpha=0.5)
  plt.plot(y_pos, line_values)

  if y_limits is not None:
    plt.ylim(y_limits)
  plt.draw()
  plt.pause(.0001)

def main():
  bar_values = [10,8,1,4,2,6]
  line_values = [10,9,7,7,5,5]
  for step in range(10):
    draw_graph('intervals', (0,50), 'values', bar_values, line_values)
    plt.pause(2)
    bar_values = bar_values[1:]+[step]
    line_values = line_values[1:]+[step*0.75]
  plt.show()
 
if __name__ == '__main__':
  main()
