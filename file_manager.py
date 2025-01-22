import uproot
import numpy as np
import pandas as pd
import awkward as ak
from tqdm import tqdm
import math
from more_itertools import batched
from urllib.request import urlopen
import yaml
import re
import glob
from pathlib import Path

CYGNO_ANALYSIS = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/"
RUN_5 = "/RECO/Run5/"
CYGNO_SIMULATION = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-sim/"
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
                        with uproot.open(f"{CYGNO_ANALYSIS}{RUN_5}reco_run{run_number}_3D.root", num_workers = 8) as root_file:
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

class Isotope:
    def __init__(self, name, dataframe, t_sim):
        self.name = name
        self.dataframe = dataframe
        self.t_sim = t_sim

class InternalBkgSource:
    def __init__(self, name, isotopes_list):
        self.name = name
        self.isotopes_list = isotopes_list

class Simulation:
    def __init__(self, int_bkg_sources_list):
        self.int_bkg_sources_list = int_bkg_sources_list

class SimulationManager:
    def __init__(self, run_number, int_bkg_sources, ext_bkg_sources, geant4_catalog):
        self.run_number = run_number
        self.int_bkg_sources = int_bkg_sources
        self.ext_bkg_sources = ext_bkg_sources
        self.geant4_catalog = geant4_catalog

    def calc_t_sim(self, N_sim_decays, mass, activity):
        return N_sim_decays/(mass * activity)

    def read_internal_bkg_data(self):
        run_file_path = f"LIME-digitized/"
        response = urlopen(f"{CYGNO_SIMULATION}")
        xml = response.read().decode('utf-8')

        geant4_catalog = pd.read_csv(self.geant4_catalog)
        
        int_bkg_sources_list = []
        
        with open('components_mass.yaml', 'r') as file:
            masses = yaml.safe_load(file)

        with open('activities.yaml', 'r') as file:
            dict_activity = yaml.full_load(file)
        
        for source in self.int_bkg_sources:
            isotopes_list = []
            folders_name = re.compile(f"{run_file_path}{source}/.*/")
            folders_list = folders_name.findall(xml)
            
            for folder in folders_list:
                isotope_name = str(folder).partition('_')[2][:-1]

                root_file_path = re.compile(f"/s3/cygno-sim/LIME_MC_data/LIME_{source}_Radioactivity_10umStep/.*_{isotope_name}.root")
                N_sim_decays = geant4_catalog[geant4_catalog["File"].str.contains(root_file_path)]["NTot"].values[0]
                isotope_activity = [name for tuple, name in dict_activity[source].get('activities').items() if isotope_name in tuple][0]
                t_sim = N_sim_decays / ( isotope_activity * masses[source] )
                
                reco_file_path = Path(f"{CYGNO_SIMULATION}{run_file_path}{source}/{folder}")
                print(list(reco_file_path.glob("*.root")))
                dataframe = uproot.open(reco_file_path.glob("reco_run*.root")[0])
                
                isotopes_list.append(Isotope(isotope_name, dataframe, t_sim))
            
            int_bkg_sources_list.append(InternalBkgSource(source, isotopes_list))
        
        return Simulation(int_bkg_sources_list)

def main():
    AmBe_campaign = [96373,98298]
    Run5_last_days = [92127,96372]

    runlog_df = pd.read_csv("runlog.csv")

    Run5 = RunManager("Run5", runlog_df, Run5_last_days[0], Run5_last_days[1])
    AmBe = RunManager("AmBe", runlog_df, AmBe_campaign[0], AmBe_campaign[1])
    
    df_list = RunManager.create_df_list(Run5)

    run_list = RunManager.add_runtype_tag(Run5, df_list)
    RunManager.merge_and_create_parquet(Run5, run_list, "Run5_data")

    internal_components = ["AcrylicBox"]
    external_components = []
    LIME_simulation = SimulationManager(5, internal_components, external_components, "geant4_catalog.csv")
    LIME_simulation.read_internal_bkg_data()

if __name__=="__main__":
    main()