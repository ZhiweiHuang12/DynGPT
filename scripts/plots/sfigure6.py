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

model_name = "on_off_nm"
model_name = "ts_stx"

observed_index =[1,2]
sample_number_li = [500,1000,2000,3000,4000]
sample_number_li = [100,200,500,1000,2000]
sample_number_li = [100,200,500,1000]

# sample_number_li = [100,500,1000,2000]
# model_name = "afl"
# observed_index =[0,1]
# # sample_number = 500
# sample_number_li = [50,100,200,300,400,500,1000]

kl_val_li  = []
observed_mean_li = []
nn_sample_mean_li = []
figure_dir = "figure/all/"
# os.makedirs(figure_dir,exist_ok=True)
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

label_values = ["Simulation","Inferred"]
width_mm = 183
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf','#1f77b4', '#ff7f0e']
colors = ["#009ACE","#D22427","#FBA933","#5AE26E","#5AE26E","#009ACE", '#ff7f0e']
# Convert width to inches (1 mm = 0.0393701 inches)
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.4  # Height is one-fourth of the width
# height_inch = width_inch * 0.1  # Height for afl

# Set the size of the figure
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(2, 5, width_ratios=[  1,1,1,1,1]) 
y_ticks_li = [[] for i in range(len(kl_val_li))]

for j in range(observed_mean_np.shape[0]):
    plot_scatter_two(fig.add_subplot(gs[0,j]), observed_mean_np[j,:,1], nn_sample_mean_np[j,:,1], label_values, "Mean",bold_point=[])
    plot_scatter_two(fig.add_subplot(gs[1,j]), observed_mean_np[j,:,2], nn_sample_mean_np[j,:,2], label_values, "Mean",bold_point=[])


plt.tight_layout()
plt.savefig(figure_dir + "sfigure6_c.jpg", dpi=400)
plt.savefig(figure_dir + "sfigure6_c.eps", dpi=400)
plt.savefig(figure_dir + "sfigure6_c.pdf", dpi=400)
plt.close()



kl_value_one = np.array(kl_val_li)[:,:,0]
kl_value_one = pd.DataFrame(kl_value_one.T,columns=sample_number_li)

kl_value_two = np.array(kl_val_li)[:,:,1]
kl_value_two = pd.DataFrame(kl_value_two.T,columns=sample_number_li)


scatter_colors = ["#B69DCA", "#A0B8DF", "#B2C4D2","#D9DBAA","#E0DFA2"]
edgecolors = ["#455EA7","#50669C","#697A86","#9EA143","#A8A837"]
width_mm=183
width_inch = width_mm * 0.0393701*0.8
height_inch = width_inch*0.5 * 0.4
fig = plt.figure(figsize=(width_inch, height_inch),constrained_layout=True)
# Defines the width ratio of each column
gs = GridSpec(1, 2, width_ratios=[ 1, 1], height_ratios=[1])

plot_boxplot(fig.add_subplot(gs[0]),kl_value_one,sample_number_li,latex_flag=False,palette_color=scatter_colors,line_colors=edgecolors,showfliers=False)
plot_boxplot(fig.add_subplot(gs[1]),kl_value_two,sample_number_li,latex_flag=False,palette_color=scatter_colors,line_colors=edgecolors,showfliers=False)


plt.tight_layout()
plt.savefig(figure_dir+"sfigure6_ab.jpg",dpi=400)
plt.savefig(figure_dir+"sfigure6_ab.pdf".format(model_name))
plt.close()



