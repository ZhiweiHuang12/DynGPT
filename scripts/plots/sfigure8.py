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
    infer_result = np.load(data_dir+'inferring_result_{}.npz'.format(model_name,model_name))
    observed_datas=infer_result["observed_datas"]
    nn_sample_datas = infer_result["nn_sample_datas"]
    # observed_datas = infer_result["observed_datas"]
    observed_datas = infer_result["synthetic_data"]

    keys_np = [key for key in infer_result.keys()]
    return nn_sample_datas,observed_datas

def calculate_margin_kl(prob_val_nn,prob_val_ssa):
    kl_div_result = []
    for i in range(len(prob_val_nn)):
        kl_div_result.append([kl_div(prob_val_nn[i][j],prob_val_ssa[i][j]) for j in range(len(prob_val_nn[0]))])
    kl_div_np = np.vstack(kl_div_result)
    return kl_div_np

# model_name = "on_off_nm"
# observed_index =[1,2]
# sample_number_li = [500,1000,2000,3000,4000]

model_name = "afl"
observed_index =[0,1]
# sample_number = 500
sample_number_li = [50,100,300,500,1000]
kl_val_li  = []
observed_mean_li = []
nn_sample_mean_li = []
figure_dir = "figure/all/".format(model_name)
for sample_number in sample_number_li:
    data_dir = "result/{}/{}/".format(model_name,sample_number)
    nn_sample_datas,observed_datas = get_data(data_dir,model_name)
    observed_mean_li.append(np.mean(observed_datas, axis=1))
    nn_sample_mean_li.append(np.mean(nn_sample_datas, axis=1))
    prob_val_nn = get_prob(nn_sample_datas[:,:,observed_index])
    prob_val_ssa = get_prob(observed_datas[:,:,observed_index])
    nm_nm_kl = calculate_margin_kl(prob_val_nn,prob_val_ssa)
    nm_nm_kl = pd.DataFrame(nm_nm_kl,columns=["Nascent RNA","Mature RNA"])
    kl_val_li.append(nm_nm_kl)

observed_mean_np = np.array(observed_mean_li)
nn_sample_mean_np = np.array(nn_sample_mean_li)


# np.mean(observed_datas, axis=1)
# np.mean(nn_sample_datas, axis=1)

width_mm = 183
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf','#1f77b4', '#ff7f0e']
colors = ["#009ACE","#D22427","#FBA933","#5AE26E","#5AE26E","#009ACE", '#ff7f0e']

# Convert width to inches (1 mm = 0.0393701 inches)
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.26  # Height is one-fourth of the width
# height_inch = width_inch * 0.1  # Height for afl

# Set the size of the figure
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(1, 9, width_ratios=[2,1, 1,  1,1,1,1,1,1]) 
y_ticks_li = [[] for i in range(len(kl_val_li))]

for i in range(len(kl_val_li)):
    species = kl_val_li[i].columns
    plot_boxplot(fig.add_subplot(gs[i]), kl_val_li[i], species, y_ticks=y_ticks_li[i], latex_flag=True, showfliers=False,palette_color=[colors[i]],linewidth=0.25,width=0.25)

for j in range(observed_mean_np.shape[0]):
    plot_scatter_two(fig.add_subplot(gs[j+2]), observed_mean_np[j,:,1], nn_sample_mean_np[j,:,1], label_values, "Mean")


# plt.tight_layout()
# plt.savefig(figure_dir + "sfigure8_a.jpg", dpi=400)
# plt.savefig(figure_dir + "sfigure8_a.eps", dpi=400)
# plt.savefig(figure_dir + "sfigure8_a.pdf", dpi=400)
plt.close()



kl_value_one = np.array(kl_val_li)[:,:,0]
kl_value_one = pd.DataFrame(kl_value_one.T,columns=sample_number_li)

kl_value_two = np.array(kl_val_li)[:,:,1]
kl_value_two = pd.DataFrame(kl_value_two.T,columns=sample_number_li)


# scatter_colors = ["#B69DCA", "#A0B8DF", "#B2C4D2","#D9DBAA","#E0DFA2"]
# edgecolors = ["#455EA7","#50669C","#697A86","#9EA143","#A8A837"]
scatter_colors = ["#C3E6F7","#C3E6F7","#C3E6F7","#C3E6F7","#C3E6F7"]
edgecolors = ['#1291C1','#1291C1','#1291C1','#1291C1','#1291C1',]

width_mm=183
width_inch = width_mm * 0.0393701
height_inch = width_mm * 0.0393701*0.5 * 0.32
fig = plt.figure(figsize=(width_inch, height_inch),constrained_layout=True)
# Defines the width ratio of each column
gs = GridSpec(1, 6, width_ratios=[ 2, 1,1,1,1,1,], height_ratios=[1])

color ="#C3E6F7",y_ticks=[],edgecolor='#1291C1'
plot_boxplot(fig.add_subplot(gs[0]),kl_value_two,sample_number_li,latex_flag=False,palette_color=scatter_colors,line_colors=edgecolors,showfliers=False)

for j in range(5):
    plot_scatter_two(fig.add_subplot(gs[j+1]), observed_mean_np[j,:,1], nn_sample_mean_np[j,:,1], label_values, "Mean",bold_point=[])

plt.tight_layout()
plt.savefig(figure_dir+"sfigure8.jpg",dpi=400)
plt.savefig(figure_dir+"sfigure8.pdf")
plt.close()



