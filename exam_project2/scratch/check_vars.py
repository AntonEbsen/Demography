import pandas as pd
import os

dta_path = r'c:\Users\Anton\Demography-2\exam_project2\data\raw\ipehd_data\ipehd_qje2009_master.dta'
df = pd.read_stata(dta_path)

cols = df.columns
for c in cols:
    if '1849' in c or 'rel' in c or 'pop' in c or 'edu' in c:
        print(c)
