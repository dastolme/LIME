import uproot
import numpy as np
import pandas as pd
import awkward as ak
from tqdm import tqdm
import math
from itertools import batched

CYGNO_ANALYSIS = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/RECO/Run5/"
CHUNK_SIZE = 500

class Run:
    def __init__(self, type, dataframe):
        self.type = type
        self.dataframe = dataframe

class RunManager:
    def __init__(self, name, runlog_df, run_start, run_end):
        self.name = name
        self.runlog_df = runlog_df
        self.run_start = run_start
        self.run_end   = run_end

    def create_df_list(self):

        param_list = ['run', 'event', 'pedestal_run', 'cmos_integral', 'cmos_mean', 'cmos_rms',
                    't_DBSCAN', 't_variables', 'lp_len', 't_pedsub', 't_saturation', 't_zerosup',
                    't_xycut', 't_rebin', 't_medianfilter', 't_noisered', 'nSc', 'sc_size', 'sc_nhits',
                    'sc_integral', 'sc_corrintegral', 'sc_rms', 'sc_energy', 'sc_pathlength',
                    'sc_theta', 'sc_length', 'sc_width', 'sc_longrms', 'sc_latrms', 'sc_lfullrms',
                    'sc_tfullrms', 'sc_lp0amplitude', 'sc_lp0prominence', 'sc_lp0fwhm', 'sc_lp0mean',
                    'sc_tp0fwhm', 'sc_xmean', 'sc_ymean', 'sc_xmax', 'sc_xmin', 'sc_ymax', 'sc_ymin',
                    'sc_pearson', 'sc_tgaussamp', 'sc_tgaussmean', 'sc_tgausssigma', 'sc_tchi2',
                    'sc_tstatus', 'sc_lgaussamp', 'sc_lgaussmean', 'sc_lgausssigma', 'sc_lchi2', 'sc_lstatus',
                    'Lime_pressure', 'Atm_pressure', 'Lime_temperature', 'Atm_temperature', 'Humidity',
                    'Mixture_Density']

        df_list = []

        print(f"Total runs: {self.run_end-self.run_start}")

        for chunk in batched(np.arange(self.run_start,self.run_end),CHUNK_SIZE):
            for run_number in tqdm(chunk):
                description = self.runlog_df["run_description"].values[0]
                if description != "garbage" and description != "Garbage":
                    try:
                        with uproot.open(f"{CYGNO_ANALYSIS}reco_run{run_number}_3D.root", num_workers = 8) as root_file:
                            df_root_file = root_file["Events"].arrays(param_list, library="ak")
                            df_list.append(ak.to_dataframe(df_root_file))
                    except FileNotFoundError as e:
                        continue
                    except TimeoutError as e:
                        print(f"Root file opening failed (run number = {run_number})")
        
        return df_list
    
    def add_runtype_tag(self, df_list):

        run_list = []

        for df in tqdm(df_list):
            dfinfo = self.runlog_df[self.runlog_df["run_number"]==df['run'].unique()[0]].copy()
            if len(dfinfo) == 0:
                continue
            if isinstance(dfinfo["stop_time"].values[0], float):
                if math.isnan(dfinfo["stop_time"].values[0]):
                    continue

            run = {"is_pedestal": dfinfo['pedestal_run'].values[0], "description": dfinfo["run_description"].values[0],
                   "source_pos": dfinfo["source_position"].values[0], "source_type": dfinfo["source_type"].values[0]}
            match run:
                case {"is_pedestal": 1}:
                    run_list.append(Run("pedestal", df))
                case {"is_pedestal": 0, "description": "Daily Calibration, parking"}:
                    run_list.append(Run("parking", df))
                case {"is_pedestal": 0, "source_pos": 3.5}:
                    run_list.append(Run("step1", df))
                case {"is_pedestal": 0, "source_pos": 10.5}:
                    run_list.append(Run("step2", df))
                case {"is_pedestal": 0, "source_pos": 17.5}:
                    run_list.append(Run("step3", df))
                case {"is_pedestal": 0, "source_pos": 24.5}:
                    run_list.append(Run("step4", df))
                case {"is_pedestal": 0, "source_pos": 32.5}:
                    run_list.append(Run("step5", df))
                case {"is_pedestal": 0, "source_type": 0}:
                    run_list.append(Run("data", df))
                case {"is_pedestal": 0, "source_type": 2}:
                    run_list.append(Run("data", df))

        return run_list
                

    def merge_and_create_parquet(self, run_list, folder):

        run_type = ["pedestal", "parking", "step1", "step2", "step3", "step4", "step5", "data"]
        
        for type in run_type:
            file_name = f"{folder}/{type}.parquet"          
            df_list = []

            for run in run_list:
                if run.type == type:
                    df_list.append(run.dataframe)
            
            if len(df_list) != 0:
                df = pd.concat(df_list)
                df.to_parquet(file_name)

class Simulation:
    def __init__(self, component, equivalent_time, dataframe):
        self.component = component
        self.equivalent_time = equivalent_time
        self.dataframe = dataframe

class SimulationManager:
    def __init__(self, components_list, geant4_catalog):
        self.components_list = components_list
        self.geant4_catalog = geant4_catalog

def main():
    AmBe_campaign = [96373,98298]
    Run5_last_days = [92127,96372]

    runlog_df = pd.read_csv("runlog.csv")

    Run5 = RunManager("Run5", runlog_df, Run5_last_days[0], Run5_last_days[1])
    AmBe = RunManager("AmBe", runlog_df, AmBe_campaign[0], AmBe_campaign[1])
    
    df_list = RunManager.create_df_list(Run5)

    run_list = RunManager.add_runtype_tag(Run5, df_list)
    RunManager.merge_and_create_parquet(Run5, run_list, "Run5_data")

if __name__=="__main__":
    main()