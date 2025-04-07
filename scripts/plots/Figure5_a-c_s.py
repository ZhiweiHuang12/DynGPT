import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gaussian_kde
import os
from matplotlib.gridspec import GridSpec
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT_pre/"
sys.path.append(root_path)
os.chdir(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *

model_name = "afl"
data_col_names = ['$k_{on}$',"$k_{off}$","$k_{syn}$","$k_{deg}$"]
model_name = "toggle_switch"
data_col_names = ['$k_{on}$',"$k_{off}$","$k_{syn}$","$k_{deg}$","1","2","3","4","5"]
model_name = "isc"
data_col_names = ["1","2","3"]

data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/result/{}/".format(model_name)
figure_dir = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/figure/"
infer_result = np.load(data_path+'inferring_result_{}.npz'.format(model_name))
es_loss,es_params =infer_result["es_loss"], infer_result["es_params"]
true_params,es_param = infer_result["true_params"],infer_result["es_param"]
scatter_color = "#C1E8FB"  #009ACE
edgecolor = "#2494BF"

# plot inference density by violin
param_index = 2

for param_index in range(es_params.shape[0]):
    sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
    true_values = true_params[param_index][:es_params.shape[2]]  # 每个小提琴图对应的真实值
    data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)

    width_mm = 183*0.3
    width_inch = width_mm * 0.0393701
    height_inch = width_inch 
    height_inch = 183*0.44* 0.0393701*0.5
    # Create grid of subplots
    fig_cols = 1
    fig, axes = plt.subplots(nrows=1, ncols=fig_cols, figsize=(width_inch, height_inch))
    plot_violin(axes,data,true_values,edgecolor = scatter_color)
    plt.tight_layout()
    plt.savefig(figure_dir + "{}/{}_posterior_violin_{}.jpg".format(model_name,model_name,param_index),dpi=300)
    plt.savefig(figure_dir + "{}/{}_posterior_violin{}.pdf".format(model_name,model_name,param_index),dpi=300)
    plt.close()
