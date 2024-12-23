import numpy as np
import pandas as pd

PIXEL_LINEAR_SIZE = 0.152 # mm

class AnalysisManager:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.dataframe_cut = dataframe

    def apply_quality_cuts(self):
        rms_quality_cut = self.dataframe['sc_rms'] > 6
        t_gausssigma_quality_cut = self.dataframe['sc_tgausssigma'] * PIXEL_LINEAR_SIZE > 0.5
        
        self.dataframe_cut = self.dataframe_cut[rms_quality_cut & t_gausssigma_quality_cut]

        return self
    
    def apply_fiducial_cuts(self, x_min, x_max, y_min, y_max):
        fiducial_cut_xmin = self.calibration_df['sc_xmin'] > x_min
        fiducial_cut_xmax = self.calibration_df['sc_xmax'] < x_max
        fiducial_cut_ymin = self.calibration_df['sc_ymin'] > y_min
        fiducial_cut_ymax = self.calibration_df['sc_ymax'] < y_max

        self.dataframe_cut = self.dataframe_cut[fiducial_cut_xmin & fiducial_cut_xmax &
                                            fiducial_cut_ymin & fiducial_cut_ymax]

        return self