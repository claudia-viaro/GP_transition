import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, expit as sigmoid
import pandas as pd




def fig1(df, name, directory):  
  file_name_multi = name +  ".png" 
  fig = plt.figure()
  x_axis = np.sort(df['Xa_reset'])
  plt.axhline(y = 0.5, color = 'r', linestyle = '-')
  plt.scatter(x_axis, df["Probability"], s=8)
  plt.xlabel("Xa_pre")
  plt.ylabel("P(y=1)")
  fig.savefig(directory + file_name_multi, dpi=300, bbox_inches='tight')
  plt.close(fig)





