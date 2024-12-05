import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PIXEL_LINEAR_SIZE = 0.152 #mm

AmBe_data = pd.read_parquet("AmBe_data.parquet")
AmBe_data['delta'] = AmBe_data['sc_integral']/AmBe_data['sc_nhits']

rms_quality_cut = AmBe_data['sc_rms'] > 6
t_gausssigma_quality_cut = AmBe_data['sc_tgausssigma'] * PIXEL_LINEAR_SIZE > 0.5
data_cut = AmBe_data[rms_quality_cut & t_gausssigma_quality_cut]
normalized_sc_length = data_cut['sc_length'] * PIXEL_LINEAR_SIZE

plt.hist2d(normalized_sc_length, data_cut['delta'],
           bins = [np.linspace(0, 200, 1000), np.linspace(0, 100, 500)], cmin = 1)
plt.xlabel('sc_length * PIXEL_LINEAR_SIZE [mm]')
plt.ylabel('delta [counts/pixel]')
plt.show()