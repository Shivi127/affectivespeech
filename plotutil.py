import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import time

def draw_graph(title, y_label, bar_values, line_values):
  intervals = range(-1*len(bar_values) + 1, 1)
  y_pos = np.arange(len(intervals))
  plt.cla()
  plt.xticks(y_pos, intervals)
  plt.ylabel(y_label)
  plt.title(title)

  plt.bar(y_pos, bar_values, align='center', alpha=0.5)
  plt.plot(y_pos, line_values)

  plt.draw()
  plt.pause(.0001)

def main():
  bar_values = [10,8,1,4,2,6]
  line_values = [10,9,7,7,5,5]
  for step in range(10):
    draw_graph('intervals', 'values', bar_values, line_values)
    plt.pause(2)
    bar_values = bar_values[1:]+[step]
    line_values = line_values[1:]+[step*0.75]
  plt.show()
 
if __name__ == '__main__':
  main()
