import numpy as np
import pandas as pd
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
from analysis_manager import AnalysisManager, SimulationAnalysisManager
from file_manager import SimulationManager

class Calibration:
    def __init__(self, data_df, calibration_df):
        self.data_df = data_df
        self.calibration_df = calibration_df

    def fit_iron_peak(self, iron_peak_window):
        analysis_calib_data = AnalysisManager(self.calibration_df)
        analysis_calib_data.apply_quality_cuts().apply_fiducial_cuts(400,1900,400,1900).apply_slimness_cut(0.8)
        
        n_bins = np.sqrt(analysis_calib_data.dataframe_cut.size)
        counts, bin_edges = np.histogram(analysis_calib_data.dataframe_cut['sc_integral'], bins = 'auto', range = (iron_peak_window[0],iron_peak_window[1]))
        bin_widths = np.diff(bin_edges)
        x = bin_edges[:-1] + (bin_widths / 2)
        lower_index_iron_window = np.abs(x - iron_peak_window[0]).argmin()
        upper_index_iron_window = np.abs(x - iron_peak_window[1]).argmin()

        iron_peak = GaussianModel(prefix='g1_')
        model = iron_peak

        g1_amplitude = np.sum([bin_widths[i] * counts[i] for i in np.arange(lower_index_iron_window,upper_index_iron_window)])
        g1_height = np.max(counts[lower_index_iron_window:upper_index_iron_window])
        g1_center = x[np.where(counts == g1_height)]
        g1_sigma = g1_amplitude / (g1_height * np.sqrt(2 * np.pi))
        params = model.make_params(g1_amplitude=g1_amplitude, g1_center=g1_center, g1_sigma=g1_sigma)

        
        result = model.fit(counts, params, x=x)
        comps = result.eval_components()
        plt.plot(x, counts, 'o', markersize = '2')
        plt.plot(x, comps['g1_'], '--', label='Gaussian component')
        plt.show()

        return result
    
class SimulationCalibration(Calibration):
    def __init__(self, MC_data_df):
        self.calibration_df = MC_data_df
        print(self.calibration_df.keys())
        self.calibration_df['sc_integral'] = self.calibration_df['sc_integral']/self.calibration_df['MC_cutexposure_energy']
        print(self.calibration_df)

    def calc_LY_per_z_slice(self, z_min, z_max, e_min, e_max):
        MC_calib_data = SimulationAnalysisManager(self.calibration_df)
        MC_calib_data.apply_MC_z_slice_cut(z_min, z_max).apply_MC_energy_cut(e_min, e_max)

        ROI_window = [250, 2_000]
        return Calibration.fit_iron_peak(self, ROI_window)

def main():
    data_df = pd.read_parquet("Run5_data/data.parquet")
    step1 = pd.read_parquet("Run5_data/step1.parquet")
    step2 = pd.read_parquet("Run5_data/step2.parquet")
    step3 = pd.read_parquet("Run5_data/step3.parquet")
    step4 = pd.read_parquet("Run5_data/step4.parquet")
    step5 = pd.read_parquet("Run5_data/step5.parquet")

    calibration_list = [step1, step2, step3, step4, step5]
    calibration_df = pd.concat(calibration_list)

    calibration = Calibration(data_df, step3)
    iron_peak_window = [1_000,8_000]
    fit_result = Calibration.fit_iron_peak(calibration, iron_peak_window)
    print(fit_result.fit_report())

    internal_components = ["DetectorBody"]
    external_components = []
    LIME_simulation = SimulationManager(5, internal_components, external_components, "geant4_catalog.csv")
    Run5_MC = LIME_simulation.read_internal_bkg_data_local()
    calib_df = LIME_simulation.create_calib_df(Run5_MC)

    MC_calibration = SimulationCalibration(calib_df)
    result = SimulationCalibration.calc_LY_per_z_slice(MC_calibration, 0, 100, 2, 10)
    print(result.fit_report())

if __name__=="__main__":
    main()