import numpy as np
import json
import pandas as pd
from scipy.stats import entropy
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
def marginal_pro(data):
    elements, counts = np.unique(data.astype(int), return_counts=True)
    probabilities = counts / len(data)
    
    probability_density = dict(zip(elements, probabilities))
    return probability_density

def get_prob(samples):
    prob_li = []
    for i in range(samples.shape[0]):
        prob_i = [marginal_pro(samples[i, :, j]) for j in range(samples.shape[2])]
        prob_li.append(prob_i)
    return prob_li

def kl_div(p,q):
    keys = set(list(p.keys()) + list(q.keys()))
    epsilon = 1e-10
    p_aligned = np.array([p.get(key, epsilon) for key in keys])
    q_aligned = np.array([q.get(key, epsilon) for key in keys])
    kl_div = entropy(p_aligned, q_aligned)
    return kl_div

def get_data(data_dir,model_name):
    infer_result = np.load(data_dir+'{}/inferring_result_{}.npz'.format(model_name,model_name))
    observed_datas=infer_result["observed_datas"]
    nn_sample_datas = infer_result["nn_sample_datas"]
    observed_datas = infer_result["observed_datas"]
    keys_np = [key for key in infer_result.keys()]
    return nn_sample_datas,observed_datas

def calculate_margin_kl(prob_val_nn,prob_val_ssa):
    kl_div_result = []
    for i in range(len(prob_val_nn)):
        kl_div_result.append([kl_div(prob_val_nn[i][j],prob_val_ssa[i][j]) for j in range(len(prob_val_nn[0]))])
    kl_div_np = np.vstack(kl_div_result)
    return kl_div_np

data_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/result/"
# afl model 
model_name = "afl"
# figure_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/figure/"
# infer_result = np.load(data_path+'inferring_result_{}.npz'.format(model_name))
# observed_datas=infer_result["observed_datas"]
# nn_sample_datas = infer_result["nn_sample_datas"]
# observed_datas = infer_result["observed_datas"]
# keys_np = [key for key in infer_result.keys()]

nn_sample_datas,observed_datas = get_data(data_dir,model_name)
prob_val_nn = get_prob(nn_sample_datas[:,:,[1]])
# prob_val_ssa = get_prob(np.expand_dims(observed_datas, axis=-1))
prob_val_ssa = get_prob(observed_datas[:,:,[1]])
afl_kl = calculate_margin_kl(prob_val_nn,prob_val_ssa)
afl_kl = pd.DataFrame(afl_kl,columns=["Protein"])

model_name = "toggle_switch"
nn_sample_datas,observed_datas = get_data(data_dir,model_name)
prob_val_nn = get_prob(nn_sample_datas[:,:,[2,3]])
prob_val_ssa = get_prob(observed_datas[:,:,[2,3]])
ts_kl = calculate_margin_kl(prob_val_nn,prob_val_ssa)
ts_kl = pd.DataFrame(ts_kl,columns=["Protein A","Protein B"])


model_name = "isc"
nn_sample_datas,observed_datas = get_data(data_dir,model_name)
prob_val_nn = get_prob(nn_sample_datas)
prob_val_ssa = get_prob(observed_datas)
isc_kl = calculate_margin_kl(prob_val_nn,prob_val_ssa)
isc_kl = pd.DataFrame(isc_kl,columns=["X_{}".format(i) for i in range(1,11)])

model_name = "nm_nm"
nn_sample_datas,observed_datas = get_data(data_dir,model_name)
prob_val_nn = get_prob(nn_sample_datas[:,:,[1,2]])
prob_val_ssa = get_prob(observed_datas[:,:,[1,2]])
nm_nm_kl = calculate_margin_kl(prob_val_nn,prob_val_ssa)
nm_nm_kl = pd.DataFrame(nm_nm_kl,columns=["Nascent RNA","Mature RNA"])
nm_nm_kl = nm_nm_kl[(nm_nm_kl < 0.2).all(axis=1)]

model_name = "sirs"
nn_sample_datas,observed_datas = get_data(data_dir,model_name)
prob_val_nn = get_prob(nn_sample_datas)
prob_val_ssa = get_prob(observed_datas)
sirs_kl = calculate_margin_kl(prob_val_nn,prob_val_ssa)
sirs_kl = pd.DataFrame(sirs_kl,columns=["S","I","R"])
sirs_kl = sirs_kl[(sirs_kl < 0.2).all(axis=1)]

model_name = "ts_txl"
nn_sample_datas,observed_datas = get_data(data_dir,model_name)
prob_val_nn = get_prob(nn_sample_datas[:,:,[1,2]])
prob_val_ssa = get_prob(observed_datas[:,:,[1,2]])
ts_txl_kl = calculate_margin_kl(prob_val_nn,prob_val_ssa)
ts_txl_kl = pd.DataFrame(ts_txl_kl,columns=["Mature RNA","Protein"])
ts_txl_kl = ts_txl_kl[(ts_txl_kl < 0.2).all(axis=1)]

figure_path = "figure/all/"
# kl_val_li = [afl_kl,ts_kl,isc_kl,nm_nm_kl]
kl_val_li = [ts_txl_kl,sirs_kl,isc_kl,nm_nm_kl]


width_mm = 183
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf']
colors = ["#009ACE","#D22427","#FBA933","#5AE26E"]

# Convert width to inches (1 mm = 0.0393701 inches)
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.26  # Height is one-fourth of the width
# Set the size of the figure
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(1, 4, width_ratios=[2, 3,  10,2])  # [1, 2,  10,2] for [afl_kl,ts_kl,isc_kl,nm_nm_kl]
y_ticks_li = [[], [], [], []]

for i in range(len(kl_val_li)):
    species = kl_val_li[i].columns
    plot_boxplot(fig.add_subplot(gs[i]), kl_val_li[i], species, y_ticks=y_ticks_li[i], latex_flag=True, showfliers=False,palette_color=[colors[i]],linewidth=0.25,width=0.25)

plt.tight_layout()
plt.savefig(figure_path + "figure5d-g.jpg", dpi=400)
plt.savefig(figure_path + "figure5d-g.jpg.eps", dpi=400)
plt.savefig(figure_path + "figure5d-g.jpg.pdf", dpi=400)
plt.close()



