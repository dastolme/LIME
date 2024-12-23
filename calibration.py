import numpy as np
import pandas as pd
from lmfit.models import GaussianModel, ExponentialModel
import matplotlib.pyplot as plt
from analysis_manager import AnalysisManager

class Calibration:
    def __init__(self, data_df, calibration_df):
        self.data_df = data_df
        self.calibration_df = calibration_df

    def fit_iron_peak(self, iron_peak_window):
        analysis_calib_data = AnalysisManager(self.calibration_df)
        analysis_calib_data.apply_quality_cuts().apply_fiducial_cuts(400,1900,400,1900).apply_slimness_cut(0.8)

        counts, bin_edges = np.histogram(analysis_calib_data.dataframe_cut['sc_integral'], bins = 600, range = (0,iron_peak_window[1]))
        bin_widths = np.diff(bin_edges)
        x = bin_edges[:-1] + (bin_widths / 2)
        lower_index_iron_window = np.abs(x - iron_peak_window[0]).argmin()
        upper_index_iron_window = np.abs(x - iron_peak_window[1]).argmin()

        background = ExponentialModel(prefix='exp_')
        iron_peak = GaussianModel(prefix='g1_')
        model = background + iron_peak

        exp_amplitude = np.max(counts[:lower_index_iron_window])
        exp_peak_index = np.where(counts == exp_amplitude)[0][0]
        half_life_index = np.abs(counts[:lower_index_iron_window] - exp_amplitude/2).argmin()
        exp_decay = np.log(2) * x[half_life_index]
        g1_amplitude = np.sum([bin_widths[i] * counts[i] for i in np.arange(lower_index_iron_window,upper_index_iron_window)])
        g1_height = np.max(counts[lower_index_iron_window:upper_index_iron_window])
        g1_center = x[np.where(counts == g1_height)]
        g1_sigma = g1_amplitude/ (g1_height * np.sqrt(2 * np.pi))
        
        params = model.make_params(g1_amplitude=g1_amplitude, g1_center=g1_center, g1_sigma=g1_sigma, 
                                   exp_amplitude=exp_amplitude, exp_decay=exp_decay)
        
        result = model.fit(counts[exp_peak_index:], params, x=x[exp_peak_index:])
        comps = result.eval_components()
        plt.plot(x[exp_peak_index:], counts[exp_peak_index:], 'o', markersize = '2')
        plt.plot(x[exp_peak_index:], comps['g1_'], '--', label='Gaussian component')
        plt.plot(x[exp_peak_index:], comps['exp_'], '--', label='Exp component')
        plt.show()

        return result

def main():
    data_df = pd.read_parquet("Run5_data/data.parquet")
    step1 = pd.read_parquet("Run5_data/step1.parquet")
    step2 = pd.read_parquet("Run5_data/step2.parquet")
    step3 = pd.read_parquet("Run5_data/step3.parquet")
    step4 = pd.read_parquet("Run5_data/step4.parquet")
    step5 = pd.read_parquet("Run5_data/step5.parquet")

    calibration_list = [step1, step2, step3, step4, step5]
    calibration_df = pd.concat(calibration_list)

    calibration = Calibration(data_df, calibration_df)
    iron_peak_window = [2_000,8_000]
    fit_result = Calibration.fit_iron_peak(calibration, iron_peak_window)
    print(fit_result.fit_report())

if __name__=="__main__":
    main()