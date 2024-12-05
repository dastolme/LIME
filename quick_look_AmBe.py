import uproot
import numpy as np
import pandas as pd
import awkward as ak

CYGNO_ANALYSIS = "https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/RECO/Run5/"

AmBe_campaign = [96373,96612]
Run5_last_days = [95792,96372]

runlog_df = pd.read_csv("runlog.csv")