import uproot
import numpy as np
import pandas as pd
import awkward as ak
from tqdm import tqdm
import math

CYGNO_ANALYSIS = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/RECO/Run5/"

AmBe_campaign = [96373,96612]
Run5_last_days = [95792,96372]

runlog_df = pd.read_csv("runlog.csv")

def create_df_list(run_start,run_end):

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

    for run_number in tqdm(np.arange(run_start,run_end)):
        if runlog_df["run_description"].values[0] != "garbage" and runlog_df["run_description"].values[0] != "Garbage":
            with uproot.open(f"{CYGNO_ANALYSIS}reco_run{run_number}_3D.root") as root_file:
                df_root_file = root_file["Events"].arrays(param_list, library="ak")
                df_list.append(ak.to_dataframe(df_root_file))
    
    return df_list

AmBe_df_list = create_df_list(AmBe_campaign[0],AmBe_campaign[1])

AmBe_data_df_list = []
AmBe_pedestal_df_list = []
AmBe_parking_df_list = []
AmBe_step1_df_list = []
AmBe_step2_df_list = []
AmBe_step3_df_list = []
AmBe_step4_df_list = []
AmBe_step5_df_list = []

for df in tqdm(AmBe_df_list):
    dfinfo = runlog_df[runlog_df["run_number"]==df['run'].unique()[0]].copy()
    if len(dfinfo) == 0:
        continue
    if isinstance(dfinfo["stop_time"].values[0], float):
        if math.isnan(dfinfo["stop_time"].values[0]):
            continue
    if "garbage" in dfinfo["run_description"].values[0]:
        continue
    if "Garbage" in dfinfo["run_description"].values[0]:
        continue
    if dfinfo['pedestal_run'].values[0]==1:
        AmBe_pedestal_df_list.append(df)
    elif dfinfo['pedestal_run'].values[0]==0 and "parking" in dfinfo["run_description"].values[0]:
        AmBe_parking_df_list.append(df)
    elif dfinfo['pedestal_run'].values[0]==0 and dfinfo["source_position"].values[0]==3.5:
        AmBe_step1_df_list.append(df)
    elif dfinfo['pedestal_run'].values[0]==0 and dfinfo["source_position"].values[0]==10.5:
        AmBe_step2_df_list.append(df)
    elif dfinfo['pedestal_run'].values[0]==0 and dfinfo["source_position"].values[0]==17.5:
        AmBe_step3_df_list.append(df)
    elif dfinfo['pedestal_run'].values[0]==0 and dfinfo["source_position"].values[0]==24.5:
        AmBe_step4_df_list.append(df)
    elif dfinfo['pedestal_run'].values[0]==0 and dfinfo["source_position"].values[0]==32.5:
        AmBe_step5_df_list.append(df)
    elif dfinfo['pedestal_run'].values[0]==0 and dfinfo["source_type"].values[0]==0:
        AmBe_data_df_list.append(df)
    else:
        continue