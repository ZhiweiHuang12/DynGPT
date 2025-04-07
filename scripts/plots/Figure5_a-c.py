import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gaussian_kde
import os
from matplotlib.gridspec import GridSpec
import sys
# root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT_pre/"
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"

sys.path.append(root_path)
os.chdir(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *
figure_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/figure/"


def judge_density(array,lst):
    # array = np.random.randint(0, 100, (7, 20))  
    # lst = [50, 20, 90, 10, 30, 70, 60]         

    row_max = array.max(axis=1)
    row_min = array.min(axis=1)

    result = [(row_min[i] <= lst[i] <= row_max[i]) for i in range(len(lst))]
    return all(result)

# afl model 
model_name = "afl"
data_col_names = ['$k_{on}$',"$k_{off}$","$k_{syn}$","$k_{deg}$"]
data_path = "result/{}/".format(model_name)
infer_result = np.load(data_path+'inferring_result_{}.npz'.format(model_name))
es_loss,es_params =infer_result["es_loss"], infer_result["es_params"]
true_params,es_param = infer_result["true_params"],infer_result["es_param"]
scatter_color = "#C1E8FB"  #009ACE
edgecolor = "#2494BF"
# plot inference density by violin
param_index = 2
for param_index in range(es_params.shape[0]):
    try:
        sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
        true_values = true_params[param_index][:es_params.shape[2]] 
        data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
        width_mm = 183*0.3
        width_inch = width_mm * 0.0393701
        height_inch = 183*0.44* 0.0393701*0.5
        # Create grid of subplots
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
        plot_violin(axes,data,true_values,edgecolor = scatter_color)
        plt.tight_layout()
        plt.savefig(figure_dir + "{}/{}_posterior_violin_{}.jpg".format(model_name,model_name,param_index),dpi=300)
        plt.savefig(figure_dir + "{}/{}_posterior_violin{}.pdf".format(model_name,model_name,param_index),dpi=300)
        plt.close()
    except Exception as e: 
        print(f"An unexpected error occurred: {e}")
        plt.close()
        continue
# toggle switch model
model_name = "toggle_switch"
data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/result/{}/".format(model_name)
infer_result = np.load(data_path+'inferring_result_{}.npz'.format(model_name),allow_pickle=True)
es_loss,es_params =infer_result["es_loss"], infer_result["es_params"]
true_params,es_param = infer_result["true_params"],infer_result["es_param"]
data_col_names = ["$k_{on1}$", "$K$", "$k_{off1}$", "$k_{on2}$", "$k_{off2}$", "$k_{syn1}$", "$k_{syn2}$", "$k_{deg1}$", "$k_{deg2}$"]
for param_index in range(es_params.shape[0]):
    try:
        weights = 1/np.exp(es_loss[param_index])
        
        # sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
        sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],weights/sum(weights))  for param_comp_index in range(len(data_col_names))]

        true_values = true_params[param_index][:es_params[0].shape[1]] 
        data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
        width_mm = 183*0.4
        width_inch = width_mm * 0.0393701
        height_inch = 183*0.44* 0.0393701*0.5
        # Create grid of subplots
        fig,  axes= plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
        plot_violin(axes,data,true_values,edgecolor = edgecolor)
        plt.tight_layout()
        plt.savefig(figure_dir + "{}/{}_posterior_violin_{}.jpg".format(model_name,model_name,param_index),dpi=300)
        plt.savefig(figure_dir + "{}/{}_posterior_violin{}.pdf".format(model_name,model_name,param_index),dpi=300)
        plt.close()
    except Exception as e:  
        print(f"An unexpected error occurred: {e}")
        plt.close()
        continue
# isc model
edgecolor = "#2494BF"
data_col_names =  ["$ρ$", "$r$", "$δ$"]
model_name = "isc"
data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/result/{}/".format(model_name)
infer_result = np.load(data_path+'inferring_result_{}.npz'.format(model_name))
es_loss,es_params =infer_result["es_loss"], infer_result["es_params"]
true_params,es_param = infer_result["true_params"],infer_result["es_param"]
for param_index in range(es_params.shape[0]):
    sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
    true_values = true_params[param_index][:es_params.shape[2]] 
    data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)

    if judge_density(np.array(sampled_values_li),true_values):
        print(param_index)
    width_mm = 183*0.2
    width_inch = width_mm * 0.0393701
    height_inch = 183*0.44* 0.0393701*0.5
    # Create grid of subplots
    fig,  axes= plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
    plot_violin(axes,data,true_values,edgecolor = edgecolor)
    plt.tight_layout()
    plt.savefig(figure_dir + "{}/{}_posterior_violin_{}.jpg".format(model_name,model_name,param_index),dpi=300)
    plt.savefig(figure_dir + "{}/{}_posterior_violin{}.pdf".format(model_name,model_name,param_index),dpi=300)
    plt.close()


# sirs model
edgecolor = "#2494BF"
data_col_names =  ["$ρ$", "$k_S$", "$k_I$", "$k_R$", "$δ_S$", "$δ_I$", "$δ_R$"]
model_name = "sirs"
data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/result/{}/".format(model_name)
infer_result = np.load(data_path+'inferring_result_{}.npz'.format(model_name),allow_pickle=True)
es_loss,es_params =infer_result["es_loss"], infer_result["es_params"]
true_params,es_param = infer_result["true_params"],infer_result["es_param"]
# es_params,true_params,es_param =np.log10(np.exp(es_params)),np.log10(np.exp(true_params)),np.log10(np.exp(es_param))

for param_index in range(es_params.shape[0]):
# for param_index in range(100):

    try:
        weights = 1/np.exp(es_loss[param_index])
        # sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
        sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],weights/sum(weights))  for param_comp_index in range(len(data_col_names))]
        true_values = true_params[param_index][:es_params[0].shape[1]] 
        data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
        if judge_density(np.array(sampled_values_li),true_values):
            print(param_index)

        width_mm = 183*0.4
        width_inch = width_mm * 0.0393701
        height_inch = 183*0.44* 0.0393701*0.5
        # Create grid of subplots
        fig,  axes= plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))


        # data=np.exp(data.iloc[:,1:])
        # true_values=np.exp(true_values[1:])

        data,true_values = data.iloc[:,1:],true_values[1:]
        plot_violin(axes,data,true_values,edgecolor = edgecolor)
        plt.tight_layout()
        plt.savefig(figure_dir + "{}/{}_posterior_violin_{}.jpg".format(model_name,model_name,param_index),dpi=300)
        plt.savefig(figure_dir + "{}/{}_posterior_violin{}.pdf".format(model_name,model_name,param_index),dpi=300)
        plt.close()
    except Exception as e:  
        print(f"An unexpected error occurred: {e}")
        plt.close()
        continue




# ts_txl model
edgecolor = "#2494BF"
data_col_names =  ["$k_{off}$", "$ k_{on}$", "$k_{synON}$", "$k_{synOFF}$", "$k_{synM}$", "$k_{degM}$", "$k_{degP}$"]
model_name = "ts_txl"
data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/result/{}/".format(model_name)
infer_result = np.load(data_path+'inferring_result_{}_.npz'.format(model_name))
es_loss,es_params =infer_result["es_loss"], infer_result["es_params"]
true_params,es_param = infer_result["true_params"],infer_result["es_param"]
es_params,true_params,es_param =np.log10(np.exp(es_params)),np.log10(np.exp(true_params)),np.log10(np.exp(es_param))

for param_index in range(es_params.shape[0]):
    sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
    true_values = true_params[param_index][:es_params.shape[2]] 
    data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
    width_mm = 183*0.4
    width_inch = width_mm * 0.0393701
    height_inch = 183*0.44* 0.0393701*0.5
    # Create grid of subplots
    fig,  axes= plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

    # data = data.iloc[:,2:]
    # true_values = true_values[2:]
    if judge_density(np.array(sampled_values_li),true_values):
        print(param_index)


    plot_violin(axes,data,true_values,edgecolor = edgecolor)
    plt.tight_layout()
    plt.savefig(figure_dir + "{}/{}_posterior_violin_{}.jpg".format(model_name,model_name,param_index),dpi=300)
    plt.savefig(figure_dir + "{}/{}_posterior_violin{}.pdf".format(model_name,model_name,param_index),dpi=300)
    plt.close()

# nm_nm model
edgecolor = "#2494BF"
data_col_names =  ["$k_{off}$", "$ r_{off}$", "$k_{on}$", "$r_{on}$","$k_{spl}$", "$r_{spl}$", "$k_{deg}$","$r_{deg}$", "$k_{syn}$"]
model_name = "nm_nm"
data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/result/{}/".format(model_name)
infer_result = np.load(data_path+'inferring_result_{}.npz'.format(model_name))
es_loss,es_params =infer_result["es_loss"], infer_result["es_params"]
true_params,es_param = infer_result["true_params"],infer_result["es_param"]
for param_index in range(es_params.shape[0]):
    sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
    true_values = true_params[param_index][:es_params.shape[2]] 
    data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
    if judge_density(np.array(sampled_values_li),true_values):
        print(param_index)
    width_mm = 183*0.4
    width_inch = width_mm * 0.0393701
    height_inch = 183*0.44* 0.0393701*0.5
    # Create grid of subplots
    fig,  axes= plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
    plot_violin(axes,data,true_values,edgecolor = edgecolor)
    
    plt.tight_layout()
    plt.savefig(figure_dir + "{}/{}_posterior_violin_{}.jpg".format(model_name,model_name,param_index),dpi=300)
    plt.savefig(figure_dir + "{}/{}_posterior_violin{}.pdf".format(model_name,model_name,param_index),dpi=300)
    plt.close()