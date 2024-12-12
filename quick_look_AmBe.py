import pandas as pd
import numpy as np
from plotnine import *

PIXEL_LINEAR_SIZE = 0.152 #mm

Run5_data_file = "Run5_data/data.parquet"
AmBe_data_file = "AmBe_data/data.parquet"

data = pd.read_parquet(Run5_data_file)
data['delta'] = data['sc_integral']/data['sc_nhits']

rms_quality_cut = data['sc_rms'] > 6
t_gausssigma_quality_cut = data['sc_tgausssigma'] * PIXEL_LINEAR_SIZE > 0.5
data_cut = data[rms_quality_cut & t_gausssigma_quality_cut]
normalized_sc_length = data_cut['sc_length'] * PIXEL_LINEAR_SIZE

print(len(data_cut['run'].unique()))

plot = (ggplot(data_cut, aes('sc_integral','delta'))
        + theme_light()
        + geom_bin_2d(bins = 200)
        + xlim(0, 100_000)
        + ylim(0, 40)
        + labs(x=None,y=None)
        )

plot.save("delta_vs_sc_integral_RUN5.png")