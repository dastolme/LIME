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

CYGNO_ANALYSIS = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/"
RUN_5 = "RECO/Run5/"
CYGNO_SIMULATION = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-sim/"
CHUNK_SIZE = 50

class RecoRun:
    def __init__(self, type, dataframe):
        self.type = type
        self.dataframe = dataframe

class RecoRunManager:
    def __init__(self, name, runlog_df, run_start, run_end):
        self.name = name
        self.runlog_df = runlog_df
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

        for chunk in batched(np.arange(self.run_start,self.run_end),CHUNK_SIZE):
            for run_number in tqdm(chunk):
                description = self.runlog_df["run_description"].values[0]
                if description != "garbage" and description != "Garbage":
                    try:
                        with uproot.open(f"{data_dir_path}{RUN_5}reco_run{run_number}_3D.root", num_workers = 8) as root_file:
                            CMOS_root_file = root_file["Events"].arrays(param_list, library="ak")
                            PMT_root_file = root_file["PMT_Events"].arrays(library="ak")
                            df_data = [ak.to_dataframe(CMOS_root_file), ak.to_dataframe(PMT_root_file)]
                            df_list.append(df_data)
                    except FileNotFoundError as e:
                        continue
                    except TimeoutError as e:
                        print(f"Root file opening failed (run number = {run_number})")
        
        return df_list
    
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
                store['CMOS'] = CMOS_df
                store['PMT'] = PMT_df

            store.close()

class Run:
    def __init__(self, run_number, CMOS_dataframe, R_PMT):
        self.run_number = run_number
        self.CMOS_dataframe = CMOS_dataframe
        self.R_PMT = R_PMT

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
        self.geant4_catalog = geant4_catalog

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
    
    def read_internal_bkg_data_local(self):
        run_file_path = f"/Users/melbadastolfo/Desktop/MC/LIME-digitized/Run{self.run_number}/"

        geant4_catalog = pd.read_csv(self.geant4_catalog)
        
        int_bkg_sources_list = []
        
        with open('components_mass.yaml', 'r') as file:
            masses = yaml.safe_load(file)

        with open('activities.yaml', 'r') as file:
            dict_activity = yaml.full_load(file)
        
        for source in self.int_bkg_sources:
            isotopes_list = []
            folders_list = [files for files in os.walk(f"{run_file_path}{source}")][0][1]
            
            for folder in folders_list:
                isotope_name = str(folder).partition('_')[2]

                root_file_path = re.compile(f"/s3/cygno-sim/LIME_MC_data/LIME_{source}_Radioactivity_10umStep/.*_{isotope_name}.root")
                N_sim_decays = geant4_catalog[geant4_catalog["File"].str.contains(root_file_path)]["NTot"].values[0]
                isotope_activity = [name for tuple, name in dict_activity[source].get('activities').items() if isotope_name in tuple][0]
                t_sim = N_sim_decays / ( isotope_activity * masses[source] )

                reco_file_path = Path(f"{run_file_path}{source}/{folder}")
                with uproot.open(list(reco_file_path.glob("*.root"))[0]) as reco_file:
                    print(reco_file['Events;1'].keys())
                    dataframe = ak.to_dataframe(reco_file['Events;1'].arrays(library = "ak"))
                
                isotopes_list.append(Isotope(isotope_name, dataframe, t_sim))

            int_bkg_sources_list.append(InternalBkgSource(source, isotopes_list))
        
        return Simulation(int_bkg_sources_list)
    
    def create_calib_df(self, Simulation):
        sources_list = Simulation.int_bkg_sources_list
        isotopes_list = [source.isotopes_list for source in sources_list]
        calib_df = pd.concat([isotope.dataframe for sublist in isotopes_list for isotope in sublist])
        
        return calib_df
        

def main():
    AmBe_campaign = [96373,96383] #98298
    Run5_last_days = [92127,96372]

    runlog_df = pd.read_csv("runlog.csv")

    Run5 = RecoRunManager("Run5", runlog_df, Run5_last_days[0], Run5_last_days[1])
    AmBe = RecoRunManager("AmBe", runlog_df, AmBe_campaign[0], AmBe_campaign[1])
    
    data_dir_path = f"{CYGNO_ANALYSIS}"
    df_list = RecoRunManager.create_df_list(AmBe, data_dir_path)

    run_list = RecoRunManager.add_runtype_tag(AmBe, df_list)
    RecoRunManager.merge_and_create_hdf5(AmBe, run_list, "AmBe_data")

    # internal_components = ["DetectorBody"]
    # external_components = []
    # LIME_simulation = SimulationManager(5, internal_components, external_components, "geant4_catalog.csv")
    # Run5_MC = LIME_simulation.read_internal_bkg_data_local()
    # LIME_simulation.create_calib_df(Run5_MC)

if __name__=="__main__":
    main()