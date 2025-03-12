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
import os
import cygno as cy
import urllib3
from concurrent.futures import ThreadPoolExecutor
import h5py
from datetime import timedelta
import dask.dataframe as dd
import aiohttp

CYGNO_ANALYSIS = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/"
RUN_5 = "RECO/Run5/"
CYGNO_SIMULATION = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-sim/"

class RecoRun:
    def __init__(self, type, dataframe):
        self.type = type
        self.dataframe = dataframe

class RecoRunManager:
    urllib3.disable_warnings()
    def __init__(self, name, run_start, run_end):
        self.name = name
        self.runlog_df = cy.read_cygno_logbook(start_run=run_start,end_run=run_end)
        self.run_start = run_start
        self.run_end   = run_end

    def create_df_list(self, data_dir_path):

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

        def read_single_file(data_dir_path, run_number):
            try:
                with uproot.open(f"{data_dir_path}reco_run{run_number}_3D.root") as root_file:
                    CMOS_root_file = root_file["Events"].arrays(param_list, library="ak")
                    PMT_root_file = root_file["PMT_Events"].arrays(library="ak")
                    df_data = [ak.to_dataframe(CMOS_root_file), ak.to_dataframe(PMT_root_file)]
                return df_data
            except FileNotFoundError as e:
                print("FileNotFound")
            except TimeoutError as e:
                print(f"Root file opening failed (run number = {run_number})")
            except aiohttp.client_exceptions.ClientResponseError as e: 
                print(f"Encountered an error: {e}")
        
        def read_many_files(run_list, data_dir_path):
            with ThreadPoolExecutor(max_workers=8) as executor:
                df_list = list(tqdm(executor.map(read_single_file, data_dir_path, run_list, chunksize=1000), total=len(run_list)))

            return df_list

        garbage_df = self.runlog_df.loc[self.runlog_df["run_description"].str.lower() == "garbage", ["run_number"]]
        run_list = [num for num in range(self.run_start,self.run_end) 
                    if garbage_df.empty or not garbage_df.isin([num]).any().any()]
        path_list = [data_dir_path for i in range(self.run_start,self.run_end)]
        df_list = read_many_files(run_list, path_list)
        
        return [df for df in df_list if df != None]
    
    def add_runtype_tag(self, df_list):

        run_list = []

        for df_data in tqdm(df_list):
            df = df_data[0]
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
                    run_list.append(RecoRun("pedestal", df_data))
                case {"is_pedestal": 0, "description": "Daily Calibration, parking"}:
                    run_list.append(RecoRun("parking", df_data))
                case {"is_pedestal": 0, "source_pos": 3.5}:
                    run_list.append(RecoRun("step1", df_data))
                case {"is_pedestal": 0, "source_pos": 10.5}:
                    run_list.append(RecoRun("step2", df_data))
                case {"is_pedestal": 0, "source_pos": 17.5}:
                    run_list.append(RecoRun("step3", df_data))
                case {"is_pedestal": 0, "source_pos": 24.5}:
                    run_list.append(RecoRun("step4", df_data))
                case {"is_pedestal": 0, "source_pos": 32.5}:
                    run_list.append(RecoRun("step5", df_data))
                case {"is_pedestal": 0, "source_type": 0}:
                    run_list.append(RecoRun("data", df_data))
                case {"is_pedestal": 0, "source_type": 2}:
                    run_list.append(RecoRun("data", df_data))

        return run_list
                

    def merge_and_create_hdf5(self, run_list, folder):

        run_type = ["pedestal", "parking", "step1", "step2", "step3", "step4", "step5", "data"]
        
        for type in run_type:
            store = pd.HDFStore(f"{folder}/{type}.h5", mode='w') 
            df_data_list = []   

            [df_data_list.append(run.dataframe) for run in run_list if run.type == type]
            
            if len(df_data_list) != 0:
                CMOS_df = pd.concat([dataframe[0] for dataframe in df_data_list])
                PMT_df = pd.concat([dataframe[1] for dataframe in df_data_list])
                store.put('CMOS', CMOS_df, format='table')
                store.put('PMT', PMT_df, format='table')

            store.close()

class Run:
    def __init__(self, run_number, CMOS_dataframe, R_PMT):
        self.run_number = run_number
        self.CMOS_dataframe = CMOS_dataframe
        self.R_PMT = R_PMT

class RunManager:
    def __init__(self, run_number, path_to_data):
        self.run_number = run_number
        self.path_to_data = path_to_data

    def read_hdf5(self): 
        return dd.read_hdf(f'{self.path_to_data}/data.h5', '/CMOS')
    
    def calc_total_runtime(self):
        urllib3.disable_warnings()
        
        df_data = RunManager.read_hdf5(self)
        runs_number = df_data["run"].unique()
        df_log = cy.read_cygno_logbook(start_run=runs_number.min(),end_run=runs_number.max()+1)
        df_log = df_log[['run_number', 'start_time', 'stop_time']]

        run_mask = df_log["run_number"].isin(runs_number)
        runs_time = df_log[run_mask]["stop_time"].sub(df_log[run_mask]["start_time"])

        return runs_time.sum().total_seconds()
    
    def calc_R_PMT(self, run_time):
        PMT_df = pd.read_hdf(f"{self.path_to_data}/data.h5", key = "PMT")
        n_wf = len(PMT_df.groupby(['entry','pmt_wf_run']).size())
        n_PMT = 4
        n_digitizer = 2

        return n_wf/n_PMT/n_digitizer/run_time

class Isotope:
    def __init__(self, name, dataframe, t_sim):
        self.name = name
        self.dataframe = dataframe
        self.t_sim = t_sim

    def __repr__(self):
        return f"Isotope name: {self.name}, Equivalent simulation time: {self.t_sim} sec \n"

class InternalBkgSource:
    def __init__(self, name, isotopes_list):
        self.name = name
        self.isotopes_list = isotopes_list

    def __repr__(self):
        return f"Component name: {self.name} \n Isotopes list: {self.isotopes_list}"

class Simulation:
    def __init__(self, int_bkg_sources_list):
        self.int_bkg_sources_list = int_bkg_sources_list

class SimulationManager:
    def __init__(self, run_number, int_bkg_sources, ext_bkg_sources, geant4_catalog):
        self.run_number = run_number
        self.int_bkg_sources = int_bkg_sources
        self.ext_bkg_sources = ext_bkg_sources
        self.geant4_catalog = pd.read_csv(geant4_catalog)

    def read_internal_bkg_data(self, file_path):

        int_bkg_sources_list = []
        
        with open('components_mass.yaml', 'r') as file:
            masses = yaml.safe_load(file)

        with open('activities.yaml', 'r') as file:
            dict_activity = yaml.full_load(file)
        
        for source in self.int_bkg_sources:
            isotopes_list = []
            source_file = h5py.File(f"{file_path}{source}.h5", 'r')
            
            for key in list(source_file.keys()):
                isotope_name = str(key).partition('_')[2][:-1]

                root_file_path = re.compile(f"/s3/cygno-sim/LIME_MC_data/LIME_{source}_Radioactivity_10umStep/.*_{isotope_name}.root")
                N_sim_decays = self.geant4_catalog[self.geant4_catalog["File"].str.contains(root_file_path)]["NTot"].values[0]
                isotope_activity = [name for tuple, name in dict_activity[source].get('activities').items() if isotope_name in tuple][0]
                t_sim = N_sim_decays / ( isotope_activity * masses[source] )
                
                dataframe = pd.read_hdf(source_file, key = f"{key}")
                
                isotopes_list.append(Isotope(isotope_name, dataframe, t_sim))
            
            int_bkg_sources_list.append(InternalBkgSource(source, isotopes_list))
        
        return Simulation(int_bkg_sources_list)
    
    def create_calib_df(self, Simulation):
        sources_list = Simulation.int_bkg_sources_list
        isotopes_list = [source.isotopes_list for source in sources_list]
        calib_df = pd.concat([isotope.dataframe for sublist in isotopes_list for isotope in sublist])
        
        return calib_df