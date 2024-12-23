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
        
        self.dataframe_cut = self.dataframe[rms_quality_cut & t_gausssigma_quality_cut]

        return self.dataframe_cut