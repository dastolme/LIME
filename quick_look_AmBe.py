import uproot
import numpy as np
import pandas as pd
import awkward as ak

CYGNO_ANALYSIS = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/RECO/Run5/"

AmBe_campaign = [96373,96612]
Run5_last_days = [95792,96372]

runlog_df = pd.read_csv("runlog.csv")

AmBe_df_list = []

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

for run_number in np.arange(AmBe_campaign[0],AmBe_campaign[1]):
    if runlog_df["run_description"].values[0] != "garbage" and runlog_df["run_description"].values[0] != "Garbage":
        with uproot.open(f"{CYGNO_ANALYSIS}reco_run{run_number}_3D.root") as root_file:
            df_root_file = root_file["Events"].arrays(param_list, library="ak")
            AmBe_df_list.append(ak.to_dataframe(df_root_file))