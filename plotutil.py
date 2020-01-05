import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import time

def draw_bar(title, y_label, values):
  intervals = range(-1*len(values) + 1, 1)
  y_pos = np.arange(len(intervals))

  plt.cla()
  plt.bar(y_pos, values, align='center', alpha=0.5)
  plt.xticks(y_pos, intervals)
  plt.ylabel(y_label)
  plt.title(title)
  plt.draw()
  plt.pause(.0001)

def main():
  values = [10,8,1,4,2,6]
  for step in range(10):
    draw_bar('intervals', 'values', values)
    values = values[1:]+[step]
    plt.pause(.2)
  plt.show()
 
if __name__ == '__main__':
  main()
