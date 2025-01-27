import pandas as pd
import numpy as np
from plotnine import *
from analysis_manager import AnalysisManager

PIXEL_LINEAR_SIZE = 0.152 # mm

Run5_data_file = "Run5_data/data.parquet"
AmBe_data_file = "AmBe_data/data.parquet"

data = pd.read_parquet(AmBe_data_file)
data['delta'] = data['sc_integral']/data['sc_nhits']

# data = data.loc[ data['run'] < 96040]
# print(len(data['run'].unique()))

analysis_run5 = AnalysisManager(data)
analysis_run5.apply_quality_cuts()
print(analysis_run5.dataframe_cut)
normalized_sc_length = analysis_run5.dataframe_cut['sc_length'] * PIXEL_LINEAR_SIZE

plot = (ggplot(analysis_run5.dataframe_cut, aes('sc_integral','delta'))
        + theme_light()
        + geom_bin_2d(bins = 200)
        + xlim(0, 100_000)
        + ylim(0, 40)
        + labs(x=None,y=None)
        )

# plot.save("delta_vs_sc_integral_AmBe.png")

Run5_step3_file = "Run5_data/step3.parquet"
step3 = pd.read_parquet(Run5_step3_file)

rms_quality_cut = step3['sc_rms'] > 6
t_gausssigma_quality_cut = step3['sc_tgausssigma'] * PIXEL_LINEAR_SIZE > 0.5
circularshape_cut = step3['sc_width'] / step3['sc_length'] > 0.75
step3_cut = step3[rms_quality_cut & t_gausssigma_quality_cut]

plot = (ggplot(step3_cut, aes('sc_integral'))
        + geom_histogram(bins = 100)
        + theme_light()
        + xlim(0, 10_000)
        + labs(x=None,y=None)
        )

# plot.save("step3.png")