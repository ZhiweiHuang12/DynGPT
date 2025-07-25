import os
import json
import numpy as np
import sys
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import pandas as pd 
import yaml
# import dyngpt
root_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
sys.path.append(root_dir)
os.chdir(root_dir)
from scripts.plots.utils_plot import *
model_name = "ts_stx"

config_name = "config2"
config_name_li = ["config{}".format(i) for i in range(2,7)]

# load loss value
loss_mean_li = []
for config_name in config_name_li:
    loss_val = np.load("result/ts_stx/{}/train_loss/fine_tune_{}_epoch900.npz".format(config_name,model_name))
    loss_mean = loss_val["loss_mean"]
    loss_mean_li.append(loss_mean)


# load kl value
kl_value_li = []
species = ["NN one","NN two","NN three","NN four","NN five"]
scatter_colors = ["#f2f2f3", "#f9e1d1", "#c6dae9","#c6dae9","#c6dae9"]
edgecolors = ["#4d4d4e","#df592a","#489bc5","#489bc5","#489bc5"]

scatter_colors = ["#B69DCA", "#A0B8DF", "#B2C4D2","#D9DBAA","#E0DFA2"]
edgecolors = ["#455EA7","#50669C","#697A86","#9EA143","#A8A837"]

data_dir = "result/"
for config_name in config_name_li:
    raw_kl_val =pd.read_csv(data_dir + "{}/{}/data/kl_{}.csv".format(model_name,config_name,model_name), sep = ',')
    kl_value_li.append(raw_kl_val)
kl_value_one = np.array(kl_value_li)[:,:,1]
kl_value_one = pd.DataFrame(kl_value_one.T,columns=species)

kl_value_two = np.array(kl_value_li)[:,:,2]
kl_value_two = pd.DataFrame(kl_value_two.T,columns=species)


def plot_loss(data,ax,colors,labels):
    for i, sublist in enumerate(data):
        ax.plot(sublist, label=labels[i],linewidth=0.5,color=colors[i])
    set_font_label(ax,"Epoch",'Loss value')
    
    ax.legend(prop={'size': 5})


# load stats value

width_mm=183
width_inch = width_mm * 0.0393701
height_inch = width_inch*0.5 * 0.4
fig = plt.figure(figsize=(width_inch, height_inch),constrained_layout=True)
# Defines the width ratio of each column
gs = GridSpec(1, 3, width_ratios=[1.5, 1, 1], height_ratios=[1])

figure_dir = "figure/all/"
plot_loss(loss_mean_li,fig.add_subplot(gs[0]),colors = edgecolors,labels=species)
plot_boxplot(fig.add_subplot(gs[1]),kl_value_one,species,latex_flag=False,palette_color=scatter_colors,line_colors=edgecolors,showfliers=False)
plot_boxplot(fig.add_subplot(gs[2]),kl_value_two,species,latex_flag=False,palette_color=scatter_colors,line_colors=edgecolors,showfliers=False)

plt.tight_layout()
plt.savefig(figure_dir+"sfigure10_a.jpg",dpi=400)
plt.savefig(figure_dir+"sfigure10_a.pdf")
plt.close()

# plot_boxplot(fig.add_subplot(gs[0,i]),kl_value_one,species,latex_flag=False,palette_color=scatter_colors,line_colors=edgecolors)

species_names = ["G",'nRNA', 'mRNA']
width_mm=183
width_inch = width_mm * 0.0393701
height_inch = width_inch*0.5 * 0.8
fig = plt.figure(figsize=(width_inch, height_inch),constrained_layout=True)

# Defines the width ratio of each column
label_values = ["SSA","DynGPT"]
gs = GridSpec(2, 5, width_ratios=[1, 1, 1,1,1], height_ratios=[1,1])

for i in range(5):
    config_name = config_name_li[i]
    stats_data_dir = data_dir + "{}/{}/data/".format(model_name,config_name)
    mean_nn_val,mean_ssa_val,max_limit = read_pair_data(stats_data_dir + "mean_nn_{}.csv".format(model_name),stats_data_dir + "mean_ssa_{}.csv".format(model_name),species_names[0])
    mean_nn_max,mean_ssa_max = mean_nn_val.max(),mean_ssa_val.max()
    mean_nn_val.columns = species_names
    mean_ssa_val.columns = species_names

    plot_scatter(fig.add_subplot(gs[0,i]),mean_nn_val,mean_ssa_val,max(mean_nn_max[1],mean_ssa_max[1]),species_names[1],label_values,"Mean",latex_flag=False,color=scatter_colors[i],edgecolor=edgecolors[i])
    plot_scatter(fig.add_subplot(gs[1,i]),mean_nn_val,mean_ssa_val,max(mean_nn_max[2],mean_ssa_max[2]),species_names[2],label_values,"Mean",latex_flag=False,color=scatter_colors[i],edgecolor=edgecolors[i])

plt.tight_layout()
plt.savefig(figure_dir+"sfigure10_b.jpg".format(model_name),dpi=400)
plt.savefig(figure_dir+"sfigure10_b.pdf".format(model_name))
plt.close()
