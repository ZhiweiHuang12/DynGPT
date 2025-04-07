import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import os
from matplotlib.gridspec import GridSpec
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
sys.path.append(root_path)
os.chdir(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *
model_type = "nm_nm"
figure_path = "figure/{}/".format(model_type)
os.makedirs(figure_path,exist_ok=True)
# train_type = "gptRL"
# data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/{}/result/{}/data/".format(train_type,model_type)
data_path = "result/{}/".format(model_type)
sub_dir = "joint_prob"

width_mm=183
#load_data 
raw_kl_val =pd.read_csv(data_path + "kl_{}.csv".format(model_type), sep = ',') 
kl_val = raw_kl_val[(raw_kl_val < 0.11).all(axis=1)]
species_names = kl_val.columns
s_i,s_j = 1,2  
mean_nn_val,mean_ssa_val,max_limit = read_pair_data(data_path + "mean_nn_{}.csv".format(model_type),data_path + "mean_ssa_{}.csv".format(model_type),species_names[0])

mean_nn_val,mean_ssa_val = mean_nn_val.iloc[kl_val.index],mean_ssa_val.iloc[kl_val.index]
mean_nn_max,mean_ssa_max = mean_nn_val.max(),mean_ssa_val.max()
std_nn_val,std_ssa_val,max_limit = read_pair_data(data_path + "std_nn_{}.csv".format(model_type),data_path + "std_ssa_{}.csv".format(model_type),species_names[0])
std_nn_val,std_ssa_val = std_nn_val.iloc[kl_val.index],std_ssa_val.iloc[kl_val.index]
std_nn_max,std_ssa_max = std_nn_val.max(),std_ssa_val.max()


cv_nn_val,cv_ssa_val = std_nn_val/mean_nn_val, std_ssa_val/mean_ssa_val
cv_nn_max,cv_ssa_max = cv_nn_val.max(),cv_ssa_val.max()

width_mm = width_mm*0.435
width_inch = width_mm * 0.0393701
height_inch = 183 *0.5* 0.0393701* 0.65  

fig = plt.figure(figsize=(width_inch, height_inch),constrained_layout=True)

y_ticks_li = [[0,25,50],[0,15,30],[],[0,25,50],[0,15,30],[]]
y_ticks_li = [[0,50,100],[0,15,30],[],[0,25,50],[0,10,20],[]]

y_ticks_li = [[] for i in range(6)]


gs = GridSpec(2, 3, width_ratios=[1, 1, 0.4], height_ratios=[1, 1])

color_scatter = "#ffe3c0"
edgecolor = "#e57a34"
# y_ticks = [0,10,20]
plot_scatter(fig.add_subplot(gs[0,0]),mean_nn_val,mean_ssa_val,max(mean_nn_max[s_i],mean_ssa_max[s_i]),species_names[s_i],label_values,"Mean",y_ticks=y_ticks_li[0],latex_flag=True,color=color_scatter,edgecolor=edgecolor)
# y_ticks = [0,5,10]
plot_scatter(fig.add_subplot(gs[0,1]),std_nn_val,std_ssa_val,max(std_nn_max[s_i],std_ssa_max[s_i]),species_names[s_i],label_values,"SD",y_ticks=y_ticks_li[1],latex_flag=True,color=color_scatter,edgecolor=edgecolor)

# y_ticks = [0,0.2,0.4,0.6,0.8]
plot_boxplot(fig.add_subplot(gs[0,2]),kl_val,species_names[s_i],y_ticks=y_ticks_li[2],latex_flag=True,palette_color = [color_scatter],line_color=edgecolor)

color_scatter = "#e7daec"
edgecolor = "#685493"
# y_ticks = [0,50,100]
plot_scatter(fig.add_subplot(gs[1,0]),mean_nn_val,mean_ssa_val,max(mean_ssa_max[s_j],mean_ssa_max[s_j]),species_names[s_j],label_values,"Mean",y_ticks=y_ticks_li[3],latex_flag=True,color=color_scatter,edgecolor=edgecolor)
# y_ticks = [0,20,40]
plot_scatter(fig.add_subplot(gs[1,1]),std_nn_val,std_ssa_val,max(std_nn_max[s_j],std_ssa_max[s_j]),species_names[s_j],label_values,"SD",y_ticks=y_ticks_li[4],latex_flag=True,color=color_scatter,edgecolor=edgecolor)
# plot_scatter(fig.add_subplot(gs[1,1]),cv_nn_val,cv_ssa_val,max(cv_nn_max[s_j],cv_ssa_max[s_j]),species_names[s_j],label_values,"SD",y_ticks=y_ticks_li[4])

# y_ticks = [0,0.2,0.4,0.6,0.8]
plot_boxplot(fig.add_subplot(gs[1,2]),kl_val,species_names[s_j],y_ticks=y_ticks_li[5],latex_flag=True,palette_color = [color_scatter],line_color=edgecolor)


plt.tight_layout()
plt.savefig(figure_path+"{}_figure4_b.jpg".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure4_b.eps".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure4_b.pdf".format(model_type),dpi=400)
plt.close()

# file_index = 22
# data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/{}/result/{}/data/joint_prob/".format(train_type,model_type)
# data1_ssa = pd.read_csv(data_path + "joint_species_ssa_counts_{}_{}_{}.csv".format(file_index[0],species_name[0],species_name[1]))
# data1_nn = pd.read_csv(data_path + "joint_species_nn_counts_{}_{}_{}.csv".format(file_index[0],species_name[0],species_name[1]))
# data1_max = data1_ssa.max()
# plot_hist_2d(axes[0, 0],data1_ssa,data1_max,"SSA",fig)

# # plot joint_prob
# 3 6 33 42 36
i=4
for i in range(0,100,3):
    file_index = [i,i+1,i+2]
    # file_index = [8,17,48]
    i1,i2,i3= 0,1,2
    sub_col_index = [1,2]
    data_ssa = pd.read_csv(data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[0]))
    sub_col = [data_ssa.columns[1],data_ssa.columns[2]]
    # data1_ssa_x_continuous = data1_ssa.iloc[:,0].values + noise1
    # data1_ssa_y_continuous = data1_ssa.iloc[:,1].values + noise2
    data1_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[i1])
    data1_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[i1])

    data2_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[i2])
    data2_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[i2])

    data3_path_ssa = data_path + "{}/joint_species_ssa_counts_{}_all.csv".format(sub_dir,file_index[i3])
    data3_path_nn = data_path + "{}/joint_species_nn_counts_{}_all.csv".format(sub_dir,file_index[i3])

    data1_nn_x_c,data1_nn_y_c,data1_ssa_x_c,data1_ssa_y_c,data1_x_lim,data1_y_lim = convert_counts_data(data1_path_ssa,data1_path_nn,sub_col_index=sub_col_index)
    data2_nn_x_c,data2_nn_y_c,data2_ssa_x_c,data2_ssa_y_c,data2_x_lim,data2_y_lim = convert_counts_data(data2_path_ssa,data2_path_nn,sub_col_index=sub_col_index)
    data3_nn_x_c,data3_nn_y_c,data3_ssa_x_c,data3_ssa_y_c,data3_x_lim,data3_y_lim = convert_counts_data(data3_path_ssa,data3_path_nn,sub_col_index=sub_col_index)

    width_mm=183*0.4
    width_inch = width_mm * 0.0393701
    height_inch = width_inch * 0.65  # 高度为宽度的四分之一
    fig, axs = plt.subplots(2, 3, figsize=(width_inch, height_inch))
    fig = plt.figure(figsize=(width_inch,height_inch))
    # gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,1], height_ratios=[1, 1])
    gs = gridspec.GridSpec(2, 3)

    cmap_name = "Blues"
    fontsize = 6
    xy_ticks = [[],[]]

    g0 = plot_jointplot(data1_nn_x_c,data1_nn_y_c,xy_ticks,axs[0, 0],lims=[data1_x_lim,data1_y_lim],label_names = sub_col,data_nn=True)
    g1 = plot_jointplot(data1_ssa_x_c,data1_ssa_y_c,xy_ticks,axs[0, 1],lims=[data1_x_lim,data1_y_lim],label_names = sub_col)

    g2 = plot_jointplot(data2_nn_x_c,data2_nn_y_c,xy_ticks,axs[0, 2],lims=[data2_x_lim,data2_y_lim],label_names = sub_col,data_nn=True)
    g3 = plot_jointplot(data2_ssa_x_c,data2_ssa_y_c,xy_ticks,axs[1, 0],lims=[data2_x_lim,data2_y_lim],label_names = sub_col)

    g4 = plot_jointplot(data3_nn_x_c,data3_nn_y_c,xy_ticks,axs[1, 1],lims=[data3_x_lim,data3_y_lim],label_names = sub_col,data_nn=True)
    g5 = plot_jointplot(data3_ssa_x_c,data3_ssa_y_c,xy_ticks,axs[1,2],lims=[data3_x_lim,data3_y_lim],label_names = sub_col)

    mg0 = SeabornFig2Grid(g1, fig, gs[0])
    mg1 = SeabornFig2Grid(g3, fig, gs[1])
    mg3 = SeabornFig2Grid(g5, fig, gs[2])
    mg4 = SeabornFig2Grid(g0, fig, gs[3])
    mg3 = SeabornFig2Grid(g2, fig, gs[4])
    mg4 = SeabornFig2Grid(g4, fig, gs[5])
    gs.tight_layout(fig)

    plt.savefig(figure_path + "{}_figure4_c_{}.jpg".format(model_type,file_index[0]))
    plt.savefig(figure_path + "{}_figure4_c_{}.pdf".format(model_type,file_index[0]),bbox_inches="tight",dpi=600)
    plt.savefig(figure_path + "{}_figure4_c_{}.eps".format(model_type,file_index[0]),bbox_inches="tight")

