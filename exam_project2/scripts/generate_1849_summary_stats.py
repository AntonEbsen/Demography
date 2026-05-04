import pandas as pd
import os

data_dir = r'c:\Users\Anton\Demography-2\exam_project2\data\raw\ipehd_data'

def load_data(filename):
    return pd.read_csv(os.path.join(data_dir, filename), encoding='latin1')

# Load the datasets
df_demo = load_data('ipehd_1849_pop_demo.csv')
df_edu = load_data('ipehd_1849_edu_stud.csv')
df_rel = load_data('ipehd_1849_rel_church.csv')
df_mari = load_data('ipehd_1849_pop_mari.csv')
df_death = load_data('ipehd_181621_pop_death.csv')

# Extract required columns
df_demo_sub = df_demo[['county', 'rb', 'pop1849_f_17to45', 'pop1849_tot']]
df_edu_sub = df_edu[['county', 'rb', 'edu1849_pub_ele_stud_m', 'edu1849_pub_ele_stud_f']]
df_rel_sub = df_rel[['county', 'rb', 'rel1849_cat_priest', 'rel1849_cat_main_church', 'rel1849_pro_main_church']]
df_mari_sub = df_mari[['county', 'rb', 'pop1849_families']]
df_death_sub = df_death[['county', 'rb', 'pop181621_born_oow_tot']]

# Merge on county and rb to handle potential key mismatch between 1800 and 1849
df_merged = df_demo_sub.merge(df_edu_sub, on=['county', 'rb'], how='left')
df_merged = df_merged.merge(df_rel_sub, on=['county', 'rb'], how='left')
df_merged = df_merged.merge(df_mari_sub, on=['county', 'rb'], how='left')
df_merged = df_merged.merge(df_death_sub, on=['county', 'rb'], how='left')

# Generate summary statistics
vars_of_interest = [
    'pop1849_f_17to45',
    'edu1849_pub_ele_stud_m',
    'edu1849_pub_ele_stud_f',
    'rel1849_cat_main_church',
    'rel1849_pro_main_church',
    'rel1849_cat_priest',
    'pop1849_families',
    'pop181621_born_oow_tot'
]

summary_stats = df_merged[vars_of_interest].describe().T
summary_stats = summary_stats[['count', 'mean', 'std', 'min', 'max']]

# Format as markdown table
md_table = summary_stats.to_markdown(floatfmt=".2f")

# Save to docs
out_dir = r'c:\Users\Anton\Demography-2\exam_project2\docs'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'summary_stats_1849.md')

with open(out_path, 'w') as f:
    f.write("# 1849 Baseline Summary Statistics\n\n")
    f.write("These variables represent the pre-Kulturkampf conditions in Prussian counties.\n\n")
    f.write("Note: Population by religion (e.g. `rel1849_cat`) was not available in the raw 1849 dataset, so `rel1849_cat_main_church` and `rel1849_pro_main_church` are included as proxies for religious density.\n\n")
    f.write(md_table)
    f.write("\n")

print("Summary statistics saved to:", out_path)
print("\nPreview:")
print(md_table)
