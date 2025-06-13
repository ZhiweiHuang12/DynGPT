import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
sys.path.append(root_path)
os.chdir(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *

data_dir = "result/"
figure_path = "figure/inference/"
os.makedirs(figure_path,exist_ok=True)
kl_val_li = []
keep_variable = [[1], [1, 2], [], []]

model_names = ["afl", "toggle_switch", "isc", "nm_nm"]
keep_variable = [[1], [2,3], [], [1,2]]
for i in range(len(model_names)):
    model_name = model_names[i]
    data_df = pd.read_csv(data_dir + "{}/kl_{}_0.1.csv".format(model_name, model_name))
    if len(keep_variable[i]) > 0:
        data_df = data_df.iloc[:, keep_variable[i]]
    data_df = data_df[(data_df < 0.2).all(axis=1)]
    print("the shape of data_df is", data_df.shape)
    kl_val_li.append(data_df)

width_mm = 183
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf']
colors = ["#009ACE","#D22427","#FBA933","#5AE26E"]

# Convert width to inches (1 mm = 0.0393701 inches)
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.26  # Height is one-fourth of the width
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(1, 4, width_ratios=[1, 2,  10,2])
y_ticks_li = [[], [], [], []]

for i in range(len(kl_val_li)):
    species = kl_val_li[i].columns
    plot_boxplot(fig.add_subplot(gs[i]), kl_val_li[i], species, y_ticks=y_ticks_li[i], latex_flag=True, showfliers=False,palette_color=[colors[i]],linewidth=0.25,width=0.25)

plt.tight_layout()
plt.savefig(figure_path + "figure5d-g.jpg", dpi=400)
plt.savefig(figure_path + "figure5d-g.jpg.eps", dpi=400)
plt.savefig(figure_path + "figure5d-g.jpg.pdf", dpi=400)
plt.close()
