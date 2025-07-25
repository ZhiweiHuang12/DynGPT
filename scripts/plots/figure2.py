import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import os
from matplotlib.gridspec import GridSpec

import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
os.chdir(root_path)
sys.path.append(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *

model_type = "sirs"
figure_path = "figure/{}/".format(model_type)
os.makedirs(figure_path,exist_ok=True)

species_name = ["S","I","R"]
figure_path = "figure/all/"
data_path = "result/{}/data/".format(model_type)
sub_dir = "joint_prob"

# plot stats
width_mm = 183*0.8
# figure_path = "figure/"
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#17becf']  
width_mm = width_mm
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.5  
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(2, 4, width_ratios=[1,1, 1, 0.6],height_ratios=[1, 1])

species_index = np.array([0,1,2])
#load_data 
raw_kl_val =pd.read_csv(data_path + "kl_{}.csv".format(model_type), sep = ',') 
kl_val = raw_kl_val[(raw_kl_val < 0.2).all(axis=1)] 
# kl_val= raw_kl_val
species_names = kl_val.columns
mean_nn_val,mean_ssa_val,max_limit = read_pair_data(data_path + "mean_nn_{}.csv".format(model_type),data_path + "mean_ssa_{}.csv".format(model_type),species_names[0])
mean_nn_val,mean_ssa_val = mean_nn_val.iloc[kl_val.index],mean_ssa_val.iloc[kl_val.index]

nn_max,ssa_max = mean_nn_val.max(),mean_ssa_val.max()
scatter_colors = ["#f2f2f3", "#f9e1d1", "#c6dae9"]
edgecolors = ["#4d4d4e","#df592a","#489bc5"]

y_ticks_li =[[0,5,10],[0,50,100],[0,30,60],[]] 
y_ticks_li =[[],[],[],[]] 

# y_ticks_li =[[],[],[],[],[]] 

for i in range(4):
    if i<3:
        species = species_names[species_index[i]]
        plot_scatter(fig.add_subplot(gs[0,i]),mean_nn_val,mean_ssa_val,max(nn_max[species],ssa_max[species]),species,label_values,"Mean",y_ticks = y_ticks_li[i],latex_flag=False,color=[scatter_colors[i]],edgecolor=edgecolors[i])
    else:
        species = species_names
        plot_boxplot(fig.add_subplot(gs[0,i]),kl_val,species,y_ticks=y_ticks_li[i],latex_flag=False,palette_color=scatter_colors,line_colors=edgecolors)

for i in range(3):
    if i<3:
        species = species_names[species_index[i]]
        plot_scatter(fig.add_subplot(gs[1,i]),mean_nn_val,mean_ssa_val,max(nn_max[species],ssa_max[species]),species,label_values,"Mean",y_ticks = y_ticks_li[i],latex_flag=True,color=[scatter_colors[i]])

plt.tight_layout()
plt.savefig(figure_path+"figure2_b.jpg",dpi=400)
plt.savefig(figure_path+"figure2_b.eps",dpi=400)
plt.savefig(figure_path+"figure2_b.pdf",dpi=400)
plt.close()


file_index = [5,6]
i1,i2= 0,1

sub_col_index_S = [0,1]
sub_col_index_R = [1,2]


data_ssa = pd.read_csv(data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[0]))
sub_col_S = [data_ssa.columns[0],data_ssa.columns[1]]
sub_col_R = [data_ssa.columns[1],data_ssa.columns[2]]

data1_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[i1])
data1_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[i1])

data2_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[i2])
data2_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[i2])



data1_nn_x_c,data1_nn_y_c,data1_ssa_x_c,data1_ssa_y_c,data1_x_lim,data1_y_lim = convert_counts_data(data1_path_ssa,data1_path_nn,sub_col_index=sub_col_index_S)
data2_nn_x_c,data2_nn_y_c,data2_ssa_x_c,data2_ssa_y_c,data2_x_lim,data2_y_lim = convert_counts_data(data1_path_ssa,data1_path_nn,sub_col_index=sub_col_index_R)
data3_nn_x_c,data3_nn_y_c,data3_ssa_x_c,data3_ssa_y_c,data3_x_lim,data3_y_lim = convert_counts_data(data2_path_ssa,data2_path_nn,sub_col_index=sub_col_index_S)
data4_nn_x_c,data4_nn_y_c,data4_ssa_x_c,data4_ssa_y_c,data4_x_lim,data4_y_lim = convert_counts_data(data2_path_ssa,data2_path_nn,sub_col_index=sub_col_index_R)

# Using Seaborn to plot the joint probability density of mRNA and protein
width_mm=183*0.7
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.45
fig, axs = plt.subplots(2, 4, figsize=(width_inch, height_inch))
fig = plt.figure(figsize=(width_inch,height_inch))
gs = gridspec.GridSpec(2, 4)

# cmap_name = "GnBu"
cmap_name = "Blues"
cmap_name = "Purples"
cmap_name = "Greens"

cmap_limit = [0.1,0.8]
fontsize = 6
# xy_ticks = [[0,2,4,6,8,10],[0,4,8,12,16]]
xy_ticks = [[],[]]
label_names_S = ["${}$".format(el) for el in sub_col_S]
label_names_R = ["${}$".format(el) for el in sub_col_R]

g0 = plot_jointplot(data1_nn_x_c,data1_nn_y_c,xy_ticks,axs[0, 0],lims=[data1_x_lim,data1_y_lim],label_names = label_names_S,data_nn=True,cmap_name=cmap_name,cmap_limit=cmap_limit)
g1 = plot_jointplot(data1_ssa_x_c,data1_ssa_y_c,xy_ticks,axs[0, 1],lims=[data1_x_lim,data1_y_lim],label_names = label_names_S,cmap_name=cmap_name,cmap_limit=cmap_limit)

g2 = plot_jointplot(data2_nn_x_c,data2_nn_y_c,xy_ticks,axs[0, 2],lims=[data2_x_lim,data2_y_lim],label_names = label_names_R,data_nn=True,cmap_name=cmap_name,cmap_limit=cmap_limit)
g3 = plot_jointplot(data2_ssa_x_c,data2_ssa_y_c,xy_ticks,axs[0, 3],lims=[data2_x_lim,data2_y_lim],label_names = label_names_R,cmap_name=cmap_name,cmap_limit=cmap_limit)

g4 = plot_jointplot(data3_nn_x_c,data3_nn_y_c,xy_ticks,axs[1, 0],lims=[data3_x_lim,data3_y_lim],label_names = label_names_S,data_nn=True,cmap_name=cmap_name,cmap_limit=cmap_limit)
g5 = plot_jointplot(data3_ssa_x_c,data3_ssa_y_c,xy_ticks,axs[1,1],lims=[data3_x_lim,data3_y_lim],label_names = label_names_S,cmap_name=cmap_name,cmap_limit=cmap_limit)

g6 = plot_jointplot(data4_nn_x_c,data4_nn_y_c,xy_ticks,axs[1, 2],lims=[data4_x_lim,data4_y_lim],label_names = label_names_R,data_nn=True,cmap_name=cmap_name,cmap_limit=cmap_limit)
g7 = plot_jointplot(data4_ssa_x_c,data4_ssa_y_c,xy_ticks,axs[1,3],lims=[data4_x_lim,data4_y_lim],label_names = label_names_R,cmap_name=cmap_name,cmap_limit=cmap_limit)

mg0 = SeabornFig2Grid(g1, fig, gs[0])
mg1 = SeabornFig2Grid(g3, fig, gs[1])
mg2 = SeabornFig2Grid(g5, fig, gs[2])
mg3 = SeabornFig2Grid(g7, fig, gs[3])
mg4 = SeabornFig2Grid(g0, fig, gs[4])
mg5 = SeabornFig2Grid(g2, fig, gs[5])
mg6 = SeabornFig2Grid(g4, fig, gs[6])
mg7 = SeabornFig2Grid(g6, fig, gs[7])
gs.tight_layout(fig)

plt.savefig(figure_path + "figure2_c_{}.jpg".format(file_index[0]))
plt.savefig(figure_path + "figure2_c_{}.pdf".format(file_index[0]),bbox_inches="tight",dpi=600)
plt.savefig(figure_path + "figure2_c_{}.eps".format(file_index[0]),bbox_inches="tight")
