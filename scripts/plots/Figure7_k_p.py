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
    kde.fit(data, sample_weight=weights)
    log_dens = kde.score_samples(data)
    densities = np.exp(log_dens)
    max_density_index = np.argmax(densities)
    max_density_point = data[max_density_index]
    return max_density_point

label_values = ["Data","DynGPT"]
model_name = "arl"
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
result_dir = root_path + "result/{}/".format(model_name)
figure_dir = root_path + "figure/inference/"

result_dir = "scripts/hyperparameter_tuning/result/arl/config6/"
temp_datas = np.load(result_dir + 'inferring_result_{}.npz'.format(model_name),allow_pickle=True)
observed_datas, nn_sample_datas, es_params, es_loss, es_param = temp_datas["observed_datas"],temp_datas["nn_sample_datas"],temp_datas["es_params"],temp_datas["es_loss"],temp_datas["es_param"]
gene_names,kl_val = temp_datas["gene_names"],temp_datas["kl_val"]

filter_index = np.where(kl_val<0.5)
observed_datas, nn_sample_datas, es_params, es_loss, es_param = observed_datas[filter_index], nn_sample_datas[filter_index], es_params[filter_index], es_loss[filter_index], es_param[filter_index]
gene_names = gene_names[filter_index]
np.corrcoef(np.mean(observed_datas, axis=1),np.mean(nn_sample_datas[:, :, 1], axis=1))
# filter_index = np.where(np.mean(observed_datas, axis=1)>100)
# observed_datas, nn_sample_datas = np.delete(observed_datas, filter_index, axis=0),np.delete(nn_sample_datas, filter_index, axis=0)

observed_data_index = [1]
x_labels = ["Burst size", "Burst frequency"]
# This part is just for drawing needs, so read the data of on_off_nm
temp_datas_nm = np.load( root_path + "result/{}/".format("on_off_nm") + 'inferring_result_{}.npz'.format("on_off_nm"),allow_pickle=True)
_, _, es_params_nm, es_loss_nm, es_param_nm, = temp_datas_nm["observed_datas"],temp_datas_nm["nn_sample_datas"],temp_datas_nm["es_params"],temp_datas_nm["es_loss"],temp_datas_nm["es_param"]

es_bs_bf = calculate_bs_bf(np.exp(es_param_nm),model_name="on_off_nm")


bold_point = [124,2545,560]
param_index_li = bold_point
width_mm = 170
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.20
param_index_li = range(0, len(es_loss))


a = np.exp(es_param[:,2])-2
param_index_li = np.where(a<0)[0]
param_index_li = np.where((a>=0) & (a<0.05))[0]
param_index_li = np.where(a>1)[0][:30]


zero_li = np.array([ np.sum(observed_datas[i, :]  == 0) for i in range(observed_datas.shape[0])])
param_index_li = np.where(zero_li>100)[0]
# param_index_li=[2548]
# param_index_li = range(0, 100)
# param_index_li = [49, 91, 133, 149, 251, 255, 262, 296, 358, 435, 445, 478, 547, 560, 586, 594, 596, 611, 617, 659, 754, 833, 1049, 1076, 1225, 1243, 1292, 1342, 1378, 1400, 1428, 1488, 1580, 1613, 1643, 1798, 1804, 1826, 1840, 2043, 2085, 2248, 2259, 2324, 2328, 2379, 2381, 2526, 2543, 2790, 2865, 2944]
for param_index in param_index_li:
    temp_es_params = np.exp(es_params_nm[0])  # 100*4 
    weights =  1 / np.exp(es_loss_nm[0])
    max_density_point = get_max_density_point(temp_es_params,weights)
    max_density_burst = calculate_bs_bf(max_density_point.reshape(1,max_density_point.shape[0]),model_name="on_off_nm")

    print("the max_density_burst is ",max_density_burst)
    plt.clf()
    # Create a new figure with specified size
    fig = plt.figure(figsize=(width_inch, height_inch))
    # Define the grid layout with 1 row and 4 columns
    gs = GridSpec(1, 4, width_ratios=[1, 1, 0.7, 1])
    for param_comp_index in range(2):
        ax = fig.add_subplot(gs[param_comp_index])

        # Calculate burst size/frequency from exponential parameters
        temp_data = np.exp(es_params_nm[0])
        values = calculate_bs_bf(temp_data,model_name="on_off_nm")[param_comp_index]
        # Density plot for estimated values
        x,y = plot_density(ax, values, weights, es_bs_bf[param_comp_index][0], x_labels[param_comp_index], label_value="Estimated value",
                     max_density_val=max_density_burst[param_comp_index][0], color='#2171A8',fill_color="#16499C")


    # Scatter plot for mean comparison between observed and neural network data
    ax = fig.add_subplot(gs[2])
    plot_scatter_two(ax, np.mean(observed_datas, axis=1), np.mean(nn_sample_datas[:, :, 1], axis=1), label_values, "Mean",bold_point=[])

    # Histogram and density plot for the first parameter
    ax = fig.add_subplot(gs[3])
    plot_hist_density(observed_datas[param_index, :], nn_sample_datas[param_index, :, observed_data_index[0]], ax, bw_method=0.15, hist_color="#CBCED3",xy_labels = ["mRNA counts","Probability"])

    # Adjust layout and save the plots as JPG and EPS
    plt.tight_layout()
    plt.savefig(figure_dir + "{}_figure7_i_l_{}.jpg".format(model_name, param_index), dpi=300)
    # plt.savefig(figure_dir + "{}_figure7_i_l_{}.eps".format(model_name, param_index), dpi=300)
    plt.savefig(figure_dir + "{}_figure7_i_l_{}.pdf".format(model_name, param_index), dpi=300)
    plt.close()


# for param_index in np.where(a>1.3)[0]:
for param_index in [124,560]:

    width_mm = 183
    width_inch = width_mm * 0.0393701
    height_inch = 170 * 0.0393701* 0.20
    es_bs_bf = calculate_bs_bf(np.exp(es_param),model_name=model_name)
    # Range of parameter indices to loop over
    param_index_li = range(0, observed_datas.shape[0])
    dist_index1,dist_index2 = 124,param_index
    # param_index = 21
    # Clear the current figure
    plt.clf()
    # Create a new figure with specified size
    fig = plt.figure(figsize=(width_inch, height_inch))

    # Define the grid layout with 1 row and 4 columns
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    # Scatter plot for mean comparison between observed and neural network data
    ax = fig.add_subplot(gs[0])
    plot_hist_density(observed_datas[dist_index1, :], nn_sample_datas[dist_index1, :, observed_data_index[0]], ax, bw_method=0.15, hist_color="#CBCED3",xy_labels = ["mRNA counts","Probability"])

    # Histogram and density plot for the first parameter
    ax = fig.add_subplot(gs[1])
    plot_hist_density(observed_datas[dist_index2, :], nn_sample_datas[dist_index2, :, observed_data_index[0]], ax, bw_method=0.15, hist_color="#CBCED3",xy_labels = ["mRNA counts","Probability"])


    temp_es_params = np.exp(es_params[param_index]) 
    weights =  1 / np.exp(es_loss[param_index])
    max_density_point = get_max_density_point(temp_es_params,weights)
    max_density_burst = calculate_bs_bf(max_density_point.reshape(1,max_density_point.shape[0]),model_name=model_name)
    # Loop over the burst size and frequency components
    for param_comp_index in range(2):
        ax = fig.add_subplot(gs[param_comp_index + 2])

        # Calculate burst size/frequency from exponential parameters
        temp_data = np.exp(es_params[param_index])
        values = calculate_bs_bf(temp_data,model_name=model_name)[param_comp_index]

        # Density plot for estimated values
        x,y = plot_density(ax, values, weights, es_bs_bf[param_comp_index][param_index], x_labels[param_comp_index], label_value="Estimated value",max_density_val=max_density_burst[param_comp_index][0])

    # Adjust layout and save the plots as JPG and EPS
    plt.tight_layout()
    plt.savefig(figure_dir + "{}_figure7_m_p_{}.jpg".format(model_name, param_index), dpi=300)
    plt.savefig(figure_dir + "{}_figure7_m_p_{}.pdf".format(model_name, param_index), dpi=300)
    plt.close()



















