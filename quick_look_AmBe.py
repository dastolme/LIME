import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PIXEL_LINEAR_SIZE = 0.152 #mm

Run5_data_file = "Run5_data/data.parquet"
AmBe_data_file = "AmBe_data/data.parquet"

data = pd.read_parquet(AmBe_data_file)
data['delta'] = data['sc_integral']/data['sc_nhits']

rms_quality_cut = data['sc_rms'] > 6
t_gausssigma_quality_cut = data['sc_tgausssigma'] * PIXEL_LINEAR_SIZE > 0.5
data_cut = data[rms_quality_cut & t_gausssigma_quality_cut]
normalized_sc_length = data_cut['sc_length'] * PIXEL_LINEAR_SIZE

print(len(data_cut['run'].unique()))

plt.hist2d(data_cut['sc_integral'], data_cut['delta'],
           bins = [np.linspace(0, 100000, 1000), np.linspace(0, 40, 500)], cmin = 1)
plt.xlabel('sc_integral')
plt.ylabel('delta [counts/pixel]')
plt.ylim(0,40)
plt.xlim(0,100000)
plt.show() 