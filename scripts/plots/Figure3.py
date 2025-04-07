import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import os
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
sys.path.append(root_path)
os.chdir(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *

def plot_compare_bar(data1,data2,labels,axi_label,ax,x_ticks=[],legend_size=6):
    data2 = data2[data2<=max(data1)]
    density1 = np.histogram(data1, bins=np.arange(min(data1), max(data1)+2), density=True)
    density2 = np.histogram(data2, bins=np.arange(min(data2), max(data2)+2), density=True)
    ax.bar(density1[1][:-1], density1[0], width=0.4, align='center', alpha=0.5, label=labels[0],color=["#C9CACA"]) ##C9CACA yellow
    ax.bar(density2[1][:-1]+0.4, density2[0], width=0.4, align='center', alpha=0.5, label=labels[1],color=["#E77A4A"]) #009ACE
    # ax.set_xlabel(axi_labels[0])
    if len(x_ticks)>0:
        ax.set_xticks(x_ticks)
    ax.set_xlim(-0.5, 20)  

    set_font_label(ax,x_label=axi_label,y_label="Probability")
    ax.legend( prop={'size':legend_size})

model_type = "isc"
# train_type = "gptRL"
# data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/{}/result/{}/data/".format(train_type,model_type)

figure_path = "figure/{}/".format(model_type)
data_path = "result/{}/".format(model_type)

os.makedirs(figure_path,exist_ok=True)


# plot stats
width_mm = 183*0.9
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#17becf']  
width_mm = width_mm*1.16
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.3 
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(2, 6, width_ratios=[1,1, 1, 1,1,1],height_ratios=[1, 1])
species_index = np.array([0,1,2,3,4,5,6,7,8,9])

#load_data 
raw_kl_val =pd.read_csv(data_path + "kl_{}.csv".format(model_type), sep = ',') 
kl_val = raw_kl_val[(raw_kl_val < 0.12).all(axis=1)] 
species_names = kl_val.columns
mean_nn_val,mean_ssa_val,max_limit = read_pair_data(data_path + "mean_nn_{}.csv".format(model_type),data_path + "mean_ssa_{}.csv".format(model_type),species_names[0])
nn_max,ssa_max = mean_nn_val.max(),mean_ssa_val.max()

y_ticks_li =[[0,5,10,15],[0,5,10],[0,4,8],[0,4,8],[],[0,3,6],[0,3,6],[0,2,4],[0,2,4],[]] 
y_ticks_li =[[0,20,40],[0,10,20],[0,8,16],[0,4,8],[],[0,3,6],[0,3,6],[0,2,4],[0,2,4],[]] 
y_ticks_li = [[] for i in range(12)]

palette_color = ["#8db2fc","#98b9f1","#a3c0e6","#aec6db","#b9cdd0","#c4d4c5","#cfdbba","#dae1af","#e5e8a4","#f0ef99"]
edgecolors = ["#4762b2","#526aa4","#5e7396","#697b88","#74837a","#808c6b","#8b945d","#969c4f","#a2a541","#adad33"]

# y_ticks_li = [[] for i in range(10)]
for i in range(6):
    if i<5:
        species = species_names[species_index[i]]
        plot_scatter(fig.add_subplot(gs[0,i]),mean_nn_val,mean_ssa_val,max(nn_max[species],ssa_max[species]),species,label_values,"Mean",y_ticks = y_ticks_li[i],color = palette_color[species_index[i]],latex_flag=True,s=6,edgecolor=edgecolors[species_index[i]])
    else:
        species = species_names[0:5]
        plot_boxplot(fig.add_subplot(gs[0,i]),kl_val,species,y_ticks=y_ticks_li[i],latex_flag=True,palette_color=palette_color[0:5], linewidth=0.25,width=0.25,line_colors=edgecolors[:5])

for i in range(6):
    if i<5:
        species = species_names[species_index[i+5]]
        plot_scatter(fig.add_subplot(gs[1,i]),mean_nn_val,mean_ssa_val,max(nn_max[species],ssa_max[species]),species,label_values,"Mean",y_ticks = y_ticks_li[i+5],color = palette_color[species_index[i+5]],latex_flag=True,s=6,edgecolor=edgecolors[species_index[i+5]])
    else:
        species = species_names[5:10]
        plot_boxplot(fig.add_subplot(gs[1,i]),kl_val,species,y_ticks=y_ticks_li[i+5],latex_flag=True,palette_color=palette_color[5:10],linewidth=0.25,width=0.25,line_colors=edgecolors[5:10])
plt.tight_layout()
plt.savefig(figure_path+"{}_figure3_c.jpg".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure3_c.eps".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure3_c.pdf".format(model_type),dpi=400)
plt.close()

width_mm = 183
width_mm = width_mm *0.7
width_inch = width_mm * 0.0393701
height_inch = width_inch *  0.4 
plt.clf()
# plt.close('all')
fig, axes = plt.subplots(1, 1, figsize=(width_inch, height_inch))
plot_boxplot(axes,kl_val,species_names,y_ticks=y_ticks_li[0],latex_flag=True,palette_color=palette_color,linewidth=0.25,width=0.25,line_colors=edgecolors)
plt.tight_layout()
plt.savefig(figure_path+"{}_figure3_b.jpg".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure3_b.eps".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure3_b.pdf".format(model_type),dpi=400)
plt.close()

# load data
for k in range(37,40,1):
    file_index = list(range(k,k+3))
    sub_dir = "joint_prob"
    data1_ssa = pd.read_csv(data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[0]))
    data2_ssa = pd.read_csv(data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[1]))
    data3_ssa = pd.read_csv(data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[2]))

    data1_nn = pd.read_csv(data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[0]))
    data2_nn = pd.read_csv(data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[1]))
    data3_nn = pd.read_csv(data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[2]))

    width_mm = 183*0.9
    width_inch = width_mm * 0.0393701
    height_inch = width_inch*0.8  
    height_inch = width_inch*0.9 

    species_index = np.array([1,2,4,5])
    cols_name = data1_ssa.columns
    fig, axes = plt.subplots(len(species_index), len(species_index), figsize=(width_inch, height_inch))
    labels = label_values
    x_ticks = [[],[0,5,10,15],[],[]]
    cmap_limit = [0.4,0.5]
    cmap_limit = [0.1,0.8]

    for i in range(len(species_index)):
        for j in range(len(species_index)):
            if i==j:
                axi_label = "${}$ counts".format(cols_name[species_index[i]])
                plot_compare_bar(data1_ssa.iloc[:,species_index[i]],data1_nn.iloc[:,species_index[i]],labels,axi_label,axes[i, j],x_ticks[i],legend_size=5)
            elif i<j:
                data_sub = data1_ssa.iloc[:,species_index[[i,j]]]
                data_max = data_sub.max()
                plot_hist_2d(axes[i, j],data_sub,data_max,"SSA",fig,False,latex_flag= True,cmap_limit=cmap_limit,xy_lim=[20,20]) # [20,18]
            elif i>j:
                data_sub = data1_nn.iloc[:,species_index[[j,i]]]
                data_max = data1_ssa.iloc[:,species_index[[j,i]]].max()
                plot_hist_2d(axes[i, j],data_sub,data_max,label_values[1],fig,False,latex_flag= True,cmap_limit=cmap_limit,xy_lim=[20,20])
    plt.tight_layout()
    plt.savefig(figure_path + "{}_figure3d_{}.jpg".format(model_type,file_index[0]),dpi=400)
    plt.savefig(figure_path + "{}_figure3d_{}.pdf".format(model_type,file_index[0]))
    plt.savefig(figure_path + "{}_figure3d_{}.eps".format(model_type,file_index[0]))

