import numpy as np
import pandas as pd

PIXEL_LINEAR_SIZE = 0.152 # mm

class AnalysisManager:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe_cut = dataframe

    def apply_quality_cuts(self):
        rms_quality_cut = self.dataframe_cut['sc_rms'] > 6
        t_gausssigma_quality_cut = self.dataframe_cut['sc_tgausssigma'] * PIXEL_LINEAR_SIZE > 0.5
        
        self.dataframe_cut = self.dataframe_cut[rms_quality_cut & t_gausssigma_quality_cut]

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

