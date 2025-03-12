import cygno as cy
import numpy as np
import pandas as pd
import dask.dataframe as dd

PIXEL_LINEAR_SIZE = 0.152 # mm

class AnalysisManager:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe_cut = dataframe
        self.oxygen_midas = pd.read_csv('/Volumes/SSD2/Data/GasSystem-Oxygen-Run5.csv')

    def apply_run_stability_cuts(self, gas_flow_min, humidity_max, oxygen_max):
        run_start = min(self.dataframe_cut['run'].unique())
        run_end = max(self.dataframe_cut['run'].unique())
        log_book = cy.read_cygno_logbook(start_run=run_start,end_run=run_end)
        
        gas_flow_cut = self.dataframe_cut['run'].isin(log_book.loc[log_book["total_gas_flow"] >= gas_flow_min, 'run_number'])
        humidity_cut = self.dataframe_cut['Humidity'] < humidity_max

        self.dataframe_cut = self.dataframe_cut[gas_flow_cut & humidity_cut]
        
        return self

    def apply_quality_cuts(self, rms_min, t_gausssigma_min, rho_min, rho_max):
        rms_quality_cut = self.dataframe_cut['sc_rms'] > rms_min
        t_gausssigma_quality_cut = self.dataframe_cut['sc_tgausssigma'] * PIXEL_LINEAR_SIZE > t_gausssigma_min
        rho_quality_lower_cut = self.dataframe_cut['sc_rms']/self.dataframe_cut['sc_nhits'] > rho_min
        rho_quality_upper_cut = self.dataframe_cut['sc_rms']/self.dataframe_cut['sc_nhits'] < rho_max
        
        self.dataframe_cut = self.dataframe_cut[rms_quality_cut & t_gausssigma_quality_cut &
                                                rho_quality_lower_cut & rho_quality_upper_cut]

        return self
    
    def apply_fiducial_cuts(self, x_min, x_max, y_min, y_max):
        fiducial_cut_xmin = self.dataframe_cut['sc_xmin'] > x_min
        fiducial_cut_xmax = self.dataframe_cut['sc_xmax'] < x_max
        fiducial_cut_ymin = self.dataframe_cut['sc_ymin'] > y_min
        fiducial_cut_ymax = self.dataframe_cut['sc_ymax'] < y_max

        self.dataframe_cut = self.dataframe_cut[fiducial_cut_xmin & fiducial_cut_xmax &
                                            fiducial_cut_ymin & fiducial_cut_ymax]

        return self
    
    def apply_slimness_cut(self, slimness_min):
        slimness_cut = self.dataframe_cut['sc_width']/self.dataframe_cut['sc_length'] > slimness_min

        self.dataframe_cut = self.dataframe_cut[slimness_cut]

        return self
    
class SimulationAnalysisManager(AnalysisManager):
    def __init__(self, dataframe):
        super().__init__(dataframe)

    def apply_MC_z_slice_cut(self, z_min, z_max):
        lower_z_cut = (self.dataframe_cut['MC_z_min'] + self.dataframe_cut['MC_z_max'])/2. > z_min
        upper_z_cut = (self.dataframe_cut['MC_z_min'] + self.dataframe_cut['MC_z_max'])/2. < z_max

        self.dataframe_cut = self.dataframe_cut[lower_z_cut & upper_z_cut]

        return self
    
    def apply_MC_energy_cut(self, e_min, e_max):
        "Energy expressed in keV"

        lower_energy_cut = self.dataframe_cut['MC_cutexposure_energy'] > e_min
        upper_energy_cut = self.dataframe_cut['MC_cutexposure_energy'] < e_max

        self.dataframe_cut = self.dataframe_cut[lower_energy_cut & upper_energy_cut]

        return self

