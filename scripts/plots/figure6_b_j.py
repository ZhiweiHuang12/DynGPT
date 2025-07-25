import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoLocator
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import math
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import os
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
sys.path.append(root_path)
os.chdir(root_path)
from dyngpt._utils._util import *
from dyngpt._utils._util_infer import *
from dyngpt._utils._util_plotting import *


def get_max_density_point(data,weights):
   
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)  
    
    kde.fit(data, sample_weight=weights) # Fitting the data using weights
    log_dens = kde.score_samples(data)
    densities = np.exp(log_dens)
    max_density_index = np.argmax(densities)
    max_density_point = data[max_density_index]
    return max_density_point

model_name = "on_off_nm"
label_values = ["Data","DynGPT"]

root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
result_dir = root_path + "result/{}/".format(model_name)
figure_dir = root_path + "figure/all/"
temp_datas = np.load(result_dir + 'inferring_result_{}.npz'.format(model_name),allow_pickle=True)

temp_datas = np.load("scripts/hyperparameter_tuning/result/ts_stx/config9/inferring_result_ts_stx.npz",allow_pickle=True)


observed_datas, nn_sample_datas, es_params, es_loss, es_param = temp_datas["observed_datas"],temp_datas["nn_sample_datas"],temp_datas["es_params"],temp_datas["es_loss"],temp_datas["es_param"]


nrna_lfc = np.mean(observed_datas[:,:,0],axis=1)/np.mean(nn_sample_datas[:,:,1],axis=1)
nrna_filter_index_mean = np.where(np.mean(observed_datas[:,:,0],axis=1)<100)[0]

nrna_filter_index  = np.where((nrna_lfc<1.5) & (nrna_lfc>0.5))[0]
mrna_lfc = np.mean(observed_datas[:,:,1],axis=1)/np.mean(nn_sample_datas[:,:,2],axis=1)
mrna_filter_index  = np.where((mrna_lfc<1.5) & (mrna_lfc>0.5))[0]
mnrna_filter_index =  list(set(nrna_filter_index) & set(mrna_filter_index) & set(nrna_filter_index_mean))
observed_datas, nn_sample_datas = observed_datas[mnrna_filter_index], nn_sample_datas[mnrna_filter_index]

param_index = "N"
width_mm = 183*0.166
width_inch = width_mm * 0.0393701
height_inch = 183*0.16*0.0393701 * 1.79 # 
param_index_li = range(0,observed_datas.shape[0])
# for param_index in range(1,100):
plt.clf()
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(2, 1)
i=param_index
color = "#AEE0EF"
ax = fig.add_subplot(gs[0,0])
observed_nrna_mean = np.mean(observed_datas[:,:,0],axis=1)
sampled_nrna_mean = np.mean(nn_sample_datas[:,:,1],axis=1)
plot_scatter_two(ax,observed_nrna_mean,sampled_nrna_mean,label_values,"Mean",color=color,edgecolor="#2A8DB4",bold_point=[0,1])

color = "#65D8EA"
ax = fig.add_subplot(gs[1,0])
observed_mrna_mean = np.mean(observed_datas[:,:,1],axis=1)
sampled_mrna_mean = np.mean(nn_sample_datas[:,:,2],axis=1)
plot_scatter_two(ax,observed_mrna_mean,sampled_mrna_mean,label_values,"Mean",color=color,edgecolor="#2A8DB4",bold_point=[0,1])

plt.tight_layout()
plt.savefig(figure_dir + "figure7b_{}_observed.jpg".format(param_index),dpi=300)
plt.savefig(figure_dir + "figure7b_{}_observed.eps".format(param_index),dpi=300)
plt.savefig(figure_dir + "figure7b_{}_observed.pdf".format(param_index),dpi=300)
plt.close()

# plot distribution comparison figure
histbins_li = [[100,40],[100,40],[40,60],[10,10],[10,10]]
xy_lim = [100,50]
histbins = histbins_li[0]
cmap_limit = [0,0.9]
observed_data_index= [1,2]
step = 4
i = 0
width_mm = 183*0.9
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.23 # 
plt.clf()
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(1, 4)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])

histbins = observed_datas[i].astype("int").max(axis=0)+1
histbins = [80,50]
plot_image_density(observed_datas[i][:,0],observed_datas[i][:,1],ax0,cmap_limit = cmap_limit,step = step,histbins=histbins)
plot_image_density(nn_sample_datas[i][:,observed_data_index][:,0],nn_sample_datas[i][:,observed_data_index][:,1],ax1,cmap_limit = cmap_limit,step = step,histbins=histbins)

i=1
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[0,3])
histbins = observed_datas[i].astype("int").max(axis=0)+1
histbins = [100,60]
plot_image_density(observed_datas[i][:,0],observed_datas[i][:,1],ax2,cmap_limit = cmap_limit,step = step,histbins=histbins)
plot_image_density(nn_sample_datas[i][:,observed_data_index][:,0],nn_sample_datas[i][:,observed_data_index][:,1],ax3,cmap_limit = cmap_limit,step = step,histbins=histbins)

plt.tight_layout()
plt.savefig(figure_dir + "figure7_c_f_{}.jpg".format(param_index),dpi=500)
plt.savefig(figure_dir + "figure7_c_f_{}.pdf".format(param_index),dpi=500)
plt.savefig(figure_dir + "figure7_c_f_{}.eps".format(param_index),dpi=500)
plt.close()

es_bs_bf = calculate_bs_bf(np.exp(es_param),model_name=model_name)
x_labels = ["Burst size", "Burst frequency"]
width_mm = 183
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.21
param_index_li = range(0, observed_datas.shape[0])
for param_index in param_index_li[:5]:

    temp_es_params = np.exp(es_params[param_index])  # 100*4 的数据
    weights =  1 / np.exp(es_loss[param_index])
    max_density_point = get_max_density_point(temp_es_params,weights)
    max_density_burst = calculate_bs_bf(max_density_point.reshape(1,max_density_point.shape[0]),model_name=model_name)
    print("the max_density_burst is ",max_density_burst)
    plt.clf()
    # Create a new figure with specified size
    fig = plt.figure(figsize=(width_inch, height_inch))
    # Define the grid layout with 1 row and 4 columns
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    # # Scatter plot for mean comparison between observed and neural network data
    # ax = fig.add_subplot(gs[0])
    # plot_scatter_two(ax, np.mean(observed_datas, axis=1), np.mean(nn_sample_datas[:, :, 1], axis=1), label_values, "Mean")

    # # Histogram and density plot for the first parameter
    # ax = fig.add_subplot(gs[1])
    # plot_hist_density(observed_datas[param_index, :], nn_sample_datas[param_index, :, observed_data_index[0]], ax, bw_method=0.15, hist_color="#CBCED3",xy_labels = ["mRNA counts","Probability"])

    # Loop over the burst size and frequency components
    for param_comp_index in range(2):
        ax = fig.add_subplot(gs[param_comp_index + 2])

        # Calculate burst size/frequency from exponential parameters
        temp_data = np.exp(es_params[param_index])
        values = calculate_bs_bf(temp_data,model_name=model_name)[param_comp_index]
        # Weighting based on loss values
        # Density plot for estimated values
        plot_density(ax, values, weights, es_bs_bf[param_comp_index][param_index], x_labels[param_comp_index], label_value="Estimated value",max_density_val=max_density_burst[param_comp_index][0])

    # Adjust layout and save the plots as JPG and EPS
    plt.tight_layout()
    plt.savefig(figure_dir + "figure7_ij_{}_observed.jpg".format(param_index), dpi=300)
    plt.savefig(figure_dir + "figure7_ij_{}_observed.eps".format( param_index), dpi=300)
    plt.close()

#-----------------figure_b_j over----------------












