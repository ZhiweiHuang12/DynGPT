import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import os
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import numpy as np 
from scipy.stats import gaussian_kde
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
sys.path.append(root_path)
os.chdir(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *

model_type = "toggle_switch"
figure_path = "figure/all/"
os.makedirs(figure_path,exist_ok=True)

# train_type = "gptRL"
# data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/{}/result/{}/data/".format(train_type,model_type)

data_path = "result/{}/".format(model_type)
sub_dir = "joint_prob"
sub_col_index = [2,3]
scatter_color_A,scatter_color_B = "#C1E8FB","#FAC0BE"
edgecolor_A,edgecolor_B = "#2494BF","#BA3334"

width_mm=183
#load_data 
raw_kl_val =pd.read_csv(data_path + "kl_{}.csv".format(model_type), sep = ',') 
kl_val = raw_kl_val[(raw_kl_val < 0.12).all(axis=1)] 
species_names = kl_val.columns
s_i,s_j = 2,3  
mean_nn_val,mean_ssa_val,max_limit = read_pair_data(data_path + "mean_nn_{}.csv".format(model_type),data_path + "mean_ssa_{}.csv".format(model_type),species_names[0])
mean_nn_val,mean_ssa_val = mean_nn_val.iloc[kl_val.index],mean_ssa_val.iloc[kl_val.index]
mean_nn_max,mean_ssa_max = mean_nn_val.max(),mean_ssa_val.max()

std_nn_val,std_ssa_val,max_limit = read_pair_data(data_path + "std_nn_{}.csv".format(model_type),data_path + "std_ssa_{}.csv".format(model_type),species_names[0])
std_nn_val,std_ssa_val = std_nn_val.iloc[kl_val.index],std_ssa_val.iloc[kl_val.index]
std_nn_max,std_ssa_max = std_nn_val.max(),std_ssa_val.max()

width_mm = width_mm*0.5
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.7 

fig = plt.figure(figsize=(width_inch, height_inch),constrained_layout=True)
gs = GridSpec(2, 3, width_ratios=[1, 1, 0.4], height_ratios=[1, 1])
y_ticks_li = [[0,10,20],[0,5,10],[0,0.2,0.4,0.6,0.8],[0,50,100],[0,20,40],[0,0.2,0.4,0.6,0.8]]
y_ticks_li = [[0,12,24],[0,4,8],[],[0,60,120],[0,15,30],[]]
y_ticks_li = [[] for i in range(6)]

plot_scatter(fig.add_subplot(gs[0,0]),mean_nn_val,mean_ssa_val,max(mean_nn_max[s_i],mean_ssa_max[s_i]),species_names[s_i],label_values,"Mean",y_ticks=y_ticks_li[0],latex_flag=True,color =scatter_color_A,edgecolor=edgecolor_A)
plot_scatter(fig.add_subplot(gs[0,1]),std_nn_val,std_ssa_val,max(std_nn_max[s_i],std_ssa_max[s_i]),species_names[s_i],label_values,"SD",y_ticks=y_ticks_li[1],latex_flag=True,color =scatter_color_A,edgecolor=edgecolor_A)

plot_boxplot(fig.add_subplot(gs[0,2]),kl_val,[species_names[s_i]],y_ticks=y_ticks_li[2],latex_flag=True,palette_color=[scatter_color_A],line_color=edgecolor_A)

plot_scatter(fig.add_subplot(gs[1,0]),mean_nn_val,mean_ssa_val,max(mean_ssa_max[s_j],mean_ssa_max[s_j]),species_names[s_j],label_values,"Mean",y_ticks=y_ticks_li[3],latex_flag=True,color =scatter_color_B,edgecolor=edgecolor_B)
plot_scatter(fig.add_subplot(gs[1,1]),std_nn_val,std_ssa_val,max(std_nn_max[s_j],std_ssa_max[s_j]),species_names[s_j],label_values,"SD",y_ticks=y_ticks_li[4],latex_flag=True,color =scatter_color_B,edgecolor=edgecolor_B)

plot_boxplot(fig.add_subplot(gs[1,2]),kl_val,[species_names[s_j]],y_ticks=y_ticks_li[5],latex_flag=True,palette_color=[scatter_color_B],line_color=edgecolor_B)

plt.tight_layout()
plt.savefig(figure_path+"sfigure2_bcde.jpg",dpi=400)
plt.savefig(figure_path+"sfigure2_bcde.eps")
plt.savefig(figure_path+"sfigure2_bcde.pdf")

# The following code is used to plot the joint probability density map in the manuscript
# file_index = [93,69,70,144]
# file_index = [96,93,70,144]
file_index = [93,70,144,145]
data_ssa = pd.read_csv(data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[0]))
sub_col = [data_ssa.columns[2],data_ssa.columns[3]]
data1_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[0])
data1_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[0])

data2_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[1])
data2_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[1])

data3_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[2])
data3_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[2])

data4_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[3])
data4_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[3])

data1_nn_x_c,data1_nn_y_c,data1_ssa_x_c,data1_ssa_y_c,data1_x_lim,data1_y_lim = convert_counts_data(data1_path_ssa,data1_path_nn,sub_col_index=sub_col_index)
data2_nn_x_c,data2_nn_y_c,data2_ssa_x_c,data2_ssa_y_c,data2_x_lim,data2_y_lim = convert_counts_data(data2_path_ssa,data2_path_nn,sub_col_index=sub_col_index)
data3_nn_x_c,data3_nn_y_c,data3_ssa_x_c,data3_ssa_y_c,data3_x_lim,data3_y_lim = convert_counts_data(data3_path_ssa,data3_path_nn,sub_col_index=sub_col_index)
data4_nn_x_c,data4_nn_y_c,data4_ssa_x_c,data4_ssa_y_c,data4_x_lim,data4_y_lim = convert_counts_data(data4_path_ssa,data4_path_nn,sub_col_index=sub_col_index)

try:
    width_mm=183*0.7
    width_inch = width_mm * 0.0393701
    height_inch = width_inch * 0.45
    fig, axs = plt.subplots(2, 4, figsize=(width_inch, height_inch))
    fig = plt.figure(figsize=(width_inch,height_inch))
    gs = gridspec.GridSpec(2, 4)

    # cmap_name = "GnBu"
    cmap_name = "Purples"
    fontsize = 6
    # xy_ticks = [[0,2,4,6,8,10],[0,4,8,12,16]]
    xy_ticks = [[],[]]
    label_names = ["${}$".format(el) for el in sub_col]
    
    g0 = plot_jointplot(data1_nn_x_c,data1_nn_y_c,xy_ticks,axs[0, 0],lims=[data1_x_lim,data1_y_lim],label_names = label_names,data_nn=True,cmap_name=cmap_name)
    g1 = plot_jointplot(data1_ssa_x_c,data1_ssa_y_c,xy_ticks,axs[0, 1],lims=[data1_x_lim,data1_y_lim],label_names = label_names,cmap_name=cmap_name)

    g2 = plot_jointplot(data2_nn_x_c,data2_nn_y_c,xy_ticks,axs[0, 2],lims=[data2_x_lim,data2_y_lim],label_names = label_names,data_nn=True,cmap_name=cmap_name)
    g3 = plot_jointplot(data2_ssa_x_c,data2_ssa_y_c,xy_ticks,axs[0, 3],lims=[data2_x_lim,data2_y_lim],label_names = label_names,cmap_name=cmap_name)

    g4 = plot_jointplot(data3_nn_x_c,data3_nn_y_c,xy_ticks,axs[1, 0],lims=[data3_x_lim,data3_y_lim],label_names = label_names,data_nn=True,cmap_name=cmap_name)
    g5 = plot_jointplot(data3_ssa_x_c,data3_ssa_y_c,xy_ticks,axs[1,1],lims=[data3_x_lim,data3_y_lim],label_names = label_names,cmap_name=cmap_name)

    g6 = plot_jointplot(data4_nn_x_c,data4_nn_y_c,xy_ticks,axs[1, 2],lims=[data4_x_lim,data4_y_lim],label_names = label_names,data_nn=True,cmap_name=cmap_name)
    g7 = plot_jointplot(data4_ssa_x_c,data4_ssa_y_c,xy_ticks,axs[1,3],lims=[data4_x_lim,data4_y_lim],label_names = label_names,cmap_name=cmap_name)

    mg0 = SeabornFig2Grid(g1, fig, gs[0])
    mg1 = SeabornFig2Grid(g3, fig, gs[1])
    mg2 = SeabornFig2Grid(g5, fig, gs[2])
    mg3 = SeabornFig2Grid(g7, fig, gs[3])
    mg4 = SeabornFig2Grid(g0, fig, gs[4])
    mg5 = SeabornFig2Grid(g2, fig, gs[5])
    mg6 = SeabornFig2Grid(g4, fig, gs[6])
    mg7 = SeabornFig2Grid(g6, fig, gs[7])
    gs.tight_layout(fig)

    plt.savefig(figure_path + "sfigure2f_{}.jpg".format(file_index[0]))
    plt.savefig(figure_path + "sfigure2f_{}.pdf".format(file_index[0]),bbox_inches="tight",dpi=600)
    plt.savefig(figure_path + "sfigure2f_{}.eps".format(file_index[0]),bbox_inches="tight")
except Exception as e:
    print("error",e)

###——————————————————figure over————————————————————————