import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import os
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from scipy.stats import kurtosis
from scipy.stats import skew
import sys
root_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/DynGPTTest/DynGPT/"
sys.append(root_path)
os.chdir(root_path)
from scripts.plots.utils_plot import *
from scripts.plots.constants import *
def calculate_probabilities(data):
    """
    Calculate the probability distribution of each integer in the two lists
    """
    total_count = len(data)
    value_counts = pd.Series(data).value_counts(normalize=True).sort_index()
    return value_counts

def get_Aligned_prob(data1,data2):
    prob1 = calculate_probabilities(data1)
    prob2 = calculate_probabilities(data2)
    df = pd.DataFrame({
        'Value': prob1.index.union(prob2.index), 
        'List1_Probability': prob1.reindex(prob1.index.union(prob2.index), fill_value=0),
        'List2_Probability': prob2.reindex(prob1.index.union(prob2.index), fill_value=0)
    })
    df = df[df["Value"]<=max(data1)]
    df['List1_Probability']=df['List1_Probability']/(df['List1_Probability'].sum())
    df['List2_Probability']=df['List2_Probability']/(df['List2_Probability'].sum())
    return df.iloc[:,[1,2]]

def plot_margin_prob(df,ax,labels,loc_pos = False,xticks=[],y_ticks=[],x_lim=0):
    df.columns=label_values
    df['Category']=list(range(df.shape[0]))
    barplot = sns.barplot(x='Category', y='value', hue='variable', data=pd.melt(df, id_vars='Category'),palette=["#C9CACA","#009ACE"], ax=ax)
    barplot.margins(x=0.1) 
    if len(xticks)>0:
        ax.set_xticks(xticks)

    if len(y_ticks)>0:
        ax.set_yticks(y_ticks)
    if x_lim>0:
        ax.set_xlim(-1,x_lim)
    legend = ax.legend(loc='upper right', fontsize='4.8', title_fontsize='6')
    set_font_label(ax,x_label=labels[0],y_label=labels[1])


# plot Bimodal coefficient
def calculate_bimodal_coefficient(data):
    m3 = skew(data, bias=False)
    m4 = kurtosis(data, fisher=True, bias=False)
    bc = 1 / (m4 + 3 - m3 ** 2)
    return bc

def calculate_bimodality_coefficient(data):
   
    n = data.shape[1]
    skewness = skew(data, axis=1, bias=False)
    kurt = kurtosis(data, axis=1, fisher=True, bias=False)  
    # bc = 1 / (kurt + 3 - skewness ** 2)
    bc = 1 / (kurt  - skewness ** 2+3)
    return bc

def plot_bimodal_coefficient(ax,x,y,z,labels=["k_off","k_syn"],cmap_limit=[0.2,0.9],cbar_flag = True):
    x, y = np.meshgrid(x, y)
    z = gaussian_filter(z, sigma=8)
    z = gaussian_filter(z, sigma=2)
    original_cmap = plt.get_cmap('viridis') #RdBu
    truncated_cmap = truncate_colormap(original_cmap, cmap_limit[0], cmap_limit[1])

    # Create a filled contour plot
    contour = ax.contourf(x, y, z, levels=12, cmap=truncated_cmap)
    contour_lines = ax.contour(x, y, z, levels=12, colors='white',linewidths=0.1)
    if cbar_flag:
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Bimodal coefficient', fontsize=6)  
        cbar.ax.tick_params(labelsize=6, length=1)  
        tick_positions = np.linspace(z.min(), z.max(), num=5)
        tick_positions = [0,0.25,0.5,0.75]
        cbar.set_ticks(tick_positions)
    set_font_label(ax,x_label=labels[0],y_label=labels[1])


font_path = '/GPUFS/sysu_jjzhang_1/hzw/data/arial.ttf'
arial_font = fm.FontProperties(fname=font_path)
rcParams['font.family'] = arial_font.get_name()
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['text.usetex'] = False

model_type = "afl"
# train_type = "gptRL"
# data_path = "/GPUFS/sysu_jjzhang_1/hzw/academicCode/metGpt2/{}/result/{}/data/".format(train_type,model_type)

data_path = "result/{}/".format(model_type)

figure_path = "figure/{}/".format(model_type)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#17becf']  
# s_i and s_j are the indices of the selected species
s_i,s_j = 0,1


width_mm = 183
scatter_color = "#C1E8FB"  #009ACE
edgecolor = "#2494BF"
width_mm = width_mm
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.19 
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(1, 6, width_ratios=[1,1, 1, 1,0.4,0.4])
col_names = ["G_u","P"]

#load_data 
kl_val =pd.read_csv(data_path + "kl_{}.csv".format(model_type), sep = ',') 

pre_species_name = kl_val.columns
kl_val.columns = col_names
species_names = kl_val.columns
mean_nn_val,mean_ssa_val,max_limit = read_pair_data(data_path + "mean_nn_{}.csv".format(model_type),data_path + "mean_ssa_{}.csv".format(model_type),pre_species_name[0])
nn_max,ssa_max = mean_nn_val.max(),mean_ssa_val.max()
mean_nn_val.columns,mean_ssa_val.columns = col_names,col_names

y_ticks = [0.0,0.4,0.8]
plot_scatter(fig.add_subplot(gs[0]),mean_nn_val,mean_ssa_val,max(nn_max[s_i],ssa_max[s_i]),species_names[s_i],label_values,"Mean",y_ticks=y_ticks,latex_flag=True,color=scatter_color,edgecolor=edgecolor)

y_ticks = [0,40,80]
plot_scatter(fig.add_subplot(gs[1]),mean_nn_val,mean_ssa_val,max(nn_max[s_j],ssa_max[s_j]),species_names[s_j],label_values,"Mean",y_ticks=y_ticks,latex_flag=True,color=scatter_color,edgecolor=edgecolor)

std_nn_val,std_ssa_val,max_limit = read_pair_data(data_path + "std_nn_{}.csv".format(model_type),data_path + "std_ssa_{}.csv".format(model_type),pre_species_name[0])
nn_max,ssa_max = std_nn_val.max(),std_ssa_val.max()
std_nn_val.columns,std_ssa_val.columns= col_names,col_names
y_ticks = [0,0.2,0.4]
plot_scatter(fig.add_subplot(gs[2]),std_nn_val,std_ssa_val,max(nn_max[s_i],ssa_max[s_i]),species_names[s_i],label_values,"SD",y_ticks=y_ticks,latex_flag=True,color=scatter_color,edgecolor=edgecolor)

y_ticks = [0,15,30]
plot_scatter(fig.add_subplot(gs[3]),std_nn_val,std_ssa_val,max(nn_max[s_j],ssa_max[s_j]),species_names[s_j],label_values,"SD",y_ticks=y_ticks,latex_flag=True,color=scatter_color,edgecolor=edgecolor)

kl_val.columns = col_names
species_names = kl_val.columns
plot_boxplot(fig.add_subplot(gs[4]),kl_val,[species_names[s_i]],latex_flag=True,showfliers=False,palette_color = [scatter_color],line_color=edgecolor)
plot_boxplot(fig.add_subplot(gs[5]),kl_val,[species_names[s_j]],latex_flag=True,showfliers=False,palette_color = [scatter_color],line_color=edgecolor)

plt.tight_layout()
plt.savefig(figure_path+"{}_figure1_bc.jpg".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure1_bc.eps".format(model_type),dpi=400)
plt.savefig(figure_path+"{}_figure1_bc.pdf".format(model_type),dpi=400)


width_mm = 183
sub_dir = "joint_prob"
data_path_ssa = data_path + "{}/joint_species_ssa_counts_all.npy".format(sub_dir)
data_path_nn = data_path + "{}/joint_species_nn_counts_all.npy".format(sub_dir)
data_ssa = np.load(data_path_ssa)[:,:,1]
data_nn = np.load(data_path_nn)[:,:,1]

data_ssa,data_nn = data_ssa.astype("int"),data_nn.astype("int")

# plot marginal probability distributions
# width_mm = width_mm *0.75
width_inch = width_mm * 0.0393701
height_inch = width_inch * 0.25  
file_list = [55,96,38,73] # 18 27 34

plt.clf()
fig = plt.figure(figsize=(width_inch, height_inch))
gs = GridSpec(1, 4, width_ratios=[1,1, 1,1])
xticks = np.arange(0, 21, 5)
y_ticks = []
labels = ["Protein counts","Probability"]
ax0 = fig.add_subplot(gs[0])
temp_prob = get_Aligned_prob(data_ssa[file_list[0],:],data_nn[file_list[0],:])
plot_margin_prob(temp_prob,ax0,labels,loc_pos=True,xticks=xticks,x_lim=22)

xticks = np.arange(0, 21, 5)
y_ticks = []
file_name = data_path + "prob_{}_1.csv".format(file_list[1])
ax1 = fig.add_subplot(gs[1])
temp_prob = get_Aligned_prob(data_ssa[file_list[1],:],data_nn[file_list[1],:])
plot_margin_prob(temp_prob,ax1,labels,loc_pos=False,xticks=xticks,x_lim=22)

xticks = np.arange(0, 81, 20)
y_ticks = []
temp_prob = get_Aligned_prob(data_ssa[file_list[2],:],data_nn[file_list[2],:])
ax2 = fig.add_subplot(gs[2])
plot_margin_prob(temp_prob,ax2,labels,loc_pos=False,xticks=xticks,y_ticks=y_ticks,x_lim=81)

xticks = np.arange(0, 81, 20)
y_ticks = []
temp_prob = get_Aligned_prob(data_ssa[file_list[3],:],data_nn[file_list[3],:])
ax3 = fig.add_subplot(gs[3])
plot_margin_prob(temp_prob,ax3,labels,loc_pos=False,xticks=xticks,y_ticks=y_ticks,x_lim=81)

plt.tight_layout()
plt.savefig(figure_path + "{}_figure1_d.jpg".format(model_type),dpi=400)
plt.savefig(figure_path + "{}_figure1_d.pdf".format(model_type))
plt.savefig(figure_path + "{}_figure1_d.eps".format(model_type))
plt.close()

# labels = ["Protein counts","Probability"]
# file_name = data_path + "prob_{}_1.csv".format(10)
# df = pd.read_csv(file_name)



# plot Bimodal coefficient
sub_dir = "joint_prob_bc"
data_path_ssa = data_path + "{}/joint_species_ssa_counts_all.npy".format(sub_dir)
data_path_nn = data_path + "{}/joint_species_nn_counts_all.npy".format(sub_dir)
data_ssa = np.load(data_path_ssa)[:,:,1]
data_nn = np.load(data_path_nn)[:,:,1]

data_ssa,data_nn = data_ssa.astype("int"),data_nn.astype("int")

df = pd.DataFrame([data_ssa[0,:],data_nn[0,:]]).T

bc_ssa = calculate_bimodality_coefficient(data_ssa)
bc_nn = calculate_bimodality_coefficient(data_nn)
correlation, p_value = pearsonr(bc_ssa, bc_nn)

x_values = np.linspace(0.01, 0.1, 100)
y_values = np.linspace(1.0, 100.0, 100)
indices = np.where(np.all(kl_val.values < 0.1, axis=1))[0] 
nn_bc_np = np.zeros((100, 100))
ssa_bc_np = np.zeros((100, 100))
stay_flag = False

index = 0
for x in range(len(x_values)):
    for y in range(len(y_values)):
        ssa_bc_np[x, y] = bc_ssa[index]
        nn_bc_np[x, y] = bc_nn[index]
        index = index + 1


width_mm = 183
width_mm = width_mm *0.5
width_inch = width_mm * 0.0393701
height_inch = width_inch *  0.4
plt.clf()
# plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(width_inch, height_inch))
# x, y = np.meshgrid(x_values, y_values)
plot_bimodal_coefficient(axes[0],x_values,y_values,ssa_bc_np.T,cbar_flag = False)
plot_bimodal_coefficient(axes[1],x_values,y_values,nn_bc_np.T,cbar_flag = False)
plt.tight_layout()
plt.savefig(figure_path + "figure1_ef.eps")
plt.savefig(figure_path + "figure1_ef.jpg",dpi=400)
plt.savefig(figure_path + "figure1_ef.pdf")
plt.close(fig)
### -----figure1 over--------------------------------