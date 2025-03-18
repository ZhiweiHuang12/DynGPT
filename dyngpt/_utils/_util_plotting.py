import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoLocator
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import math
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import matplotlib.font_manager as fm
from matplotlib import rcParams
from sklearn.neighbors import KernelDensity





def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n))
    )
    return new_cmap

def plot_hist_2d(ax, data,data_max, label_val,fig,face_color=True,latex_flag=True,cmap_val="Blues",cmap_limit=[0,0.7],xy_lim=[],vmax=0.025,vmin=0.025):
    col_names = data.columns.tolist()
    if len(xy_lim)>0:
        data = data[data[col_names[0]]<= xy_lim[0]]
        data = data[data[col_names[1]]<= xy_lim[1]]
    else:
        data = data[data[col_names[0]]<= data_max[0]]
        data = data[data[col_names[1]]<= data_max[1]]
    data_max = data.max().astype("int")
    # hist = ax.hist2d(data[col_names[0]], data[col_names[1]], bins=(data_max.values), cmap='viridis',density=True,norm=mpl.colors.LogNorm(vmax=0.025),)
    original_cmap = plt.get_cmap(cmap_val)
    truncated_cmap = truncate_colormap(original_cmap, cmap_limit[0], cmap_limit[1])
    
    # hist = ax.hist2d(data[col_names[0]], data[col_names[1]], bins=(data_max.values), cmap=truncated_cmap,density=True,norm=mpl.colors.LogNorm(vmax=0.025),)
    # hist = ax.hist2d(data[col_names[0]], data[col_names[1]], bins=(data_max.values), cmap=truncated_cmap,density=True,norm=mpl.colors.LogNorm(vmin=0.001,vmax=0.01),)
    hist = ax.hist2d(data[col_names[0]], data[col_names[1]], bins=(data_max.values), cmap=truncated_cmap,density=True,norm=mpl.colors.LogNorm(vmax=0.025),)
    
    # hist = ax.hist2d(data[col_names[0]], data[col_names[1]], bins=(data_max.values), cmap=truncated_cmap,density=True,norm=mpl.colors.LogNorm(vmax=0.025),)
    # hist = ax.hist2d(data[col_names[0]], data[col_names[1]], bins=(data_max.values), cmap=truncated_cmap,density=True,norm=mpl.colors.LogNorm(vmax=0.2),)

    if latex_flag:
        x_label = "{} ({})".format("${}$".format(col_names[0]),label_val)
        y_label = "{} ({})".format("${}$".format(col_names[1]),label_val)
    else:
        x_label = "{} ({})".format(col_names[0],label_val)
        y_label = "{} ({})".format(col_names[1],label_val)
    if len(xy_lim)>0:         
        ax.set_xlim(0, xy_lim[0])  
        ax.set_ylim(0, xy_lim[1])  
    else:
        ax.set_xlim(0, data_max[0])  
        ax.set_ylim(0, data_max[1])       
    set_font_label(ax,x_label=x_label,y_label=y_label)
    # ax.set_facecolor('lightblue')
    ax.set_facecolor('white')

    
    if face_color:
        cbar1 = fig.colorbar(hist[3], ax=ax)
        cbar1.ax.tick_params(labelsize=6)

def read_pair_data(data1_path,data2_path,column_name):
    data1 = pd.read_csv(data1_path)
    data2 = pd.read_csv(data2_path)
    x_max = data1[column_name].max()
    y_max = data2[column_name].max()
    max_limit = max(x_max, y_max)
    return data1,data2,max_limit

# def set_font_label(ax,x_label,y_label,font_size=7,font_name='Arial'):
#     ax.set_xlabel(x_label, fontsize=font_size, fontname=font_name,labelpad=1)
#     ax.set_ylabel(y_label, fontsize=font_size, fontname=font_name,labelpad=1)
#     ax.tick_params(axis='both', which='major', labelsize=6, length=1, pad=1)
#     ax.tick_params(axis='both', which='minor', labelsize=6, length=1,pad=1)
#     return ax

def set_font_label(ax,x_label,y_label,font_size=6,font_name='Arial',labelsize=5,tick_width=0.4): # 6.5
    ax.set_xlabel(x_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.set_ylabel(y_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, length=1.6, pad=1,width=tick_width)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize, length=1.6,pad=1,width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return ax


def set_limit(ax,x_lim,y_lim):
    ax.set_xlim((0,x_lim))
    ax.set_ylim((0,y_lim))

def plot_boxplot(ax,data,species_name,y_ticks=[],latex_flag=False,showfliers=True,palette_color = ['#17becf']):
    # boxprops=dict(linewidth=0.5)
    boxplot = sns.boxplot(data=data[species_name], ax=ax,width=0.2, fliersize=1,palette=palette_color,linewidth=0.5,showfliers=showfliers)
    
    if len(palette_color)==1:
        plt.setp(boxplot.artists, color=palette_color[0])  
        plt.setp(boxplot.lines, color=palette_color[0]) 
    else:
        for artist, color in zip(boxplot.artists, palette_color):
            artist.set_edgecolor(color)
            # artist.set_linewidth(0.5)  
        
        palette_color_new = [color for color in palette_color for _ in range(6)]

        for line, color in zip(boxplot.lines, palette_color_new):  # 每个boxplot有6条线（2 whiskers, 2 caps, and 2 medians）
            line.set_color(color)
            # line.set_linewidth(0.5)
    if latex_flag:
        labels = ["${}$".format(el) for el in species_name]
        ax.set_xticks(ticks=range(len(species_name)), labels=labels)
    if isinstance(species_name, list):
        set_font_label(ax,x_label="Species",y_label="KL distance")
    else:
        # set_font_label(ax,x_label=species_name,y_label="KL distance")
        set_font_label(ax,x_label="Species",y_label="KL distance")
    if len(y_ticks)>0:
        ax.set_yticks(y_ticks)
    

# '#91CAE8','grey'
def plot_scatter(ax, data1,data2,data_max, species_name,label_vals,stas_type,color = "#000000",y_ticks=[],latex_flag=False,s=2):
    ax.plot([0, data_max], [0, data_max], linestyle='--', color='grey',linewidth=0.5) #'#CF5142'
    if latex_flag:
        label_val = "$"+species_name+"$"
        sns.scatterplot( x=data1[species_name],color=color, y=data2[species_name], label= label_val ,s=s,ax=ax,linewidth=0.1,edgecolor='skyblue') # s=8
    else:
        sns.scatterplot( x=data1[species_name],color=color, y=data2[species_name], label= species_name ,s=s,ax=ax,linewidth=0.1,edgecolor='skyblue')# s=8
    ax.legend(loc='upper left', prop={'size': 6},bbox_to_anchor=(0, 0.9, 0.1, 0.1))
    
    x_label = "{} ({})".format(stas_type,label_vals[0])
    y_label = "{} ({})".format(stas_type,label_vals[1])
    ax.set_xlim(0, data_max*1.05)  
    ax.set_ylim(0, data_max*1.05) 
    if len(y_ticks)>0:
        ax.set_yticks(y_ticks)
        ax.set_xticks(y_ticks)

    set_font_label(ax,x_label=x_label,y_label=y_label)
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 

def plot_scatter_two(ax, data1,data2,label_vals,stas_type,color = '#F8DCBC',y_ticks=[],edgecolor="#D27439",linewidth2=0.5,bold_point=[]):
    max_val = max(max(data1), max(data2))*1.05
    min_val = min(min(data1), min(data2))*0.95
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='grey',linewidth=0.5)
    sns.scatterplot( x=data1,color=color,edgecolor=edgecolor, y=data2 ,s=8,ax=ax)
    if len(bold_point)>0:
        ax.scatter(data1[bold_point],data2[bold_point],s=8,edgecolor='red',facecolor='red', linewidth=linewidth2)
    # ax.legend(loc='upper left', prop={'size': 6},bbox_to_anchor=(0, 0.9, 0.1, 0.1))
    x_label = "{} ({})".format(stas_type,label_vals[0])
    y_label = "{} ({})".format(stas_type,label_vals[1])
    ax.set_xlim(min_val, max_val*1.05)  
    ax.set_ylim(min_val, max_val*1.05) 
    if len(y_ticks)>0:
        ax.set_yticks(y_ticks)
        ax.set_xticks(y_ticks)
    set_font_label(ax,x_label=x_label,y_label=y_label)
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False) 


def get_max_density_point(data,weights):
    if len(data.shape)<2:
        data=data.reshape(-1,1)
    weights =weights/sum(weights)
    # Initialize the kernel density estimator
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
    # Fitting the data using weights
    kde.fit(data, sample_weight=weights)
    log_dens = kde.score_samples(data)
    densities = np.exp(log_dens)
    max_density_index = np.argmax(densities)
    max_density_point = data[max_density_index]
    return max_density_point

def plot_density(ax,values,weights,x_position,x_label,label_value = 'Estimated value',vline_flag = True,mean_max_flag=True,max_density_val=2,linewidth=0.75, color='#E61F19',v_color='#E184FF',fill_color="#EBAC80"):
    kde = gaussian_kde(values, weights=weights)
    x = np.linspace(min(values)*0.8, max(values), 1000)
    y = kde(x)
    ax.plot(x, y,linewidth=linewidth, color=color)
    ax.fill_between(x, y, color=fill_color,alpha=0.05)
    if vline_flag:
        max_density_val = get_max_density_point(values,weights)
        ax.axvline(x=x_position, color='red', linestyle='--', label="Max density",linewidth=linewidth)

        # ax.axvline(x=x_position, color='red', linestyle='--', label=label_value)
        # if mean_max_flag:
        #     mean_val = sum(x*y/sum(y))
        #     max_val =  x[np.argmax(y)]
        #     ax.axvline(x=mean_val, color='blue', linestyle='--', label="Mean")
        #     ax.axvline(x=max_val, color='green', linestyle='--', label="max")
        # ax.axvline(x=max_density_val, color=v_color, linestyle='--', label="Max density",linewidth=linewidth)
        ax.legend(prop={'size': 5})
    set_font_label(ax,x_label,'Posterior prob.')
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False)
    return x,y



class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def plot_jointplot(data1,data2,xy_ticks,ax,lims,label_names,fontsize = 6,cmap_name = "Blues",fill_flag=True,color="#d37166",data_nn=False):
    if data_nn:
        label_names = [el + " (DynGPT)" for el in label_names]
    else:
        label_names = [el + " (SSA)" for el in label_names]
    cmap_name = "Blues"
    # g0 = sns.jointplot(x=data1, y=data2, kind="hist",  fill=fill_flag,cmap=cmap_name, color=color,ax=ax,cbar=True)
    g0 = sns.jointplot(x=data1, y=data2, kind="kde",  fill=fill_flag,cmap=cmap_name, color=color,ax=ax,cbar=True)
    if len(lims)>0:
        g0.ax_joint.set_xlim(0, lims[0]+2)
        g0.ax_joint.set_ylim(0, lims[1]+2)
    g0.ax_joint.set_xlabel(label_names[0], fontsize=fontsize,labelpad=1)
    g0.ax_joint.set_ylabel(label_names[1], fontsize=fontsize,labelpad=1)
    length = 1
    g0.ax_joint.tick_params(axis='both', which='major', labelsize=fontsize, length=length,pad=1)
    g0.ax_joint.tick_params(axis='both', which='minor', labelsize=fontsize, length=length,pad=1)
    g0.ax_marg_x.tick_params(axis='x', which='major', labelsize=6, length=length)  
    g0.ax_marg_y.tick_params(axis='y', which='major', labelsize=6, length=length) 
    if len(xy_ticks[0])>0:
        g0.ax_joint.set_xticks(xy_ticks[0])
        g0.ax_joint.set_yticks(xy_ticks[1])
    # plt.colorbar(g0.ax_joint.collections[0], ax=g0.ax_joint,location='right')
    return g0


def convert_counts_data(file_path1,file_path2,sub_col_index=[1,4]):
    data1_ssa = pd.read_csv(file_path1)
    data1_nn = pd.read_csv(file_path2)
    sub_col = [data1_ssa.columns[sub_col_index[0]],data1_ssa.columns[sub_col_index[1]]]
    data1_ssa = data1_ssa[sub_col][:10000]
    data1_nn = data1_nn[sub_col][:10000]
    data1 =  data1_nn.iloc[:,0].values 
    noise1 = np.random.uniform(0, 1, size=len(data1))
    noise2 = np.random.uniform(0, 1, size=len(data1))
    data1_nn_x_continuous = data1_nn.iloc[:,0].values + noise1
    data1_nn_y_continuous = data1_nn.iloc[:,1].values + noise2
    data1_ssa_x_continuous = data1_ssa.iloc[:,0].values + noise1
    data1_ssa_y_continuous = data1_ssa.iloc[:,1].values + noise2
    x_lim = math.ceil(max(data1_ssa_x_continuous))
    y_lim = math.ceil(max(data1_ssa_y_continuous))
    return data1_nn_x_continuous,data1_nn_y_continuous,data1_ssa_x_continuous,data1_ssa_y_continuous,x_lim,y_lim

def weighted_sample(values,weights):
    # values = es_params[param_index][:,param_comp_index]
    # weights = 1/es_loss[param_index] 
    kde = gaussian_kde(values, weights=weights)
    values = np.linspace(min(values)*0.8, max(values), 1000)
    probabilities = kde(values)
    probabilities /= probabilities.sum() 

    sampled_values = np.random.choice(values, size=1500, p=probabilities)
    return sampled_values

def plot_violin(ax,data,true_values,color="#C9CACA",edgecolor="skyblue"):
    # temp_df = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
    sns.violinplot(data=data, ax=ax,inner=None,color=color,edgecolor=edgecolor)
    ax.tick_params(axis='both', which='major', labelsize=6)
    for j, true_value in enumerate(true_values):
        if j==0:
            ax.axhline(y=true_value, xmin=j/len(true_values), xmax=(j+1)/len(true_values), color='grey', linestyle='--',label='True value')
        else:
            ax.axhline(y=true_value, xmin=j/len(true_values), xmax=(j+1)/len(true_values), color='grey', linestyle='--')

    set_font_label(ax, "Parameters", "Value")
    ax.legend(loc='upper left', prop={'size': 6},bbox_to_anchor=(0, 0.9, 0.1, 0.1))


def plot_hist_density(data1,data2,ax,xy_labels = ["Counts","Probability"],bw_method=0.15,density_label = 'DynGPT',hist_color = "b",linewidth=0.75,color="#E61F19"):
    ax.hist(data1, bins=int(max(data1))+1, density=True, alpha=0.6, color=hist_color, label='Data')
    kde = gaussian_kde(data2,bw_method=0.15)
    x_vals = np.linspace(0, max(data1),int(max(data1)))
    kde_vals = kde(x_vals)
    kde_vals = kde_vals/sum(kde_vals)
    ax.plot(x_vals, kde_vals, linewidth=linewidth, color=color, label=density_label)
    x_label,y_label = xy_labels
    set_font_label(ax,x_label=x_label,y_label=y_label)
    ax.legend(loc='upper right', prop={'size': 6})
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False)  
# def calculate_bs_bf(data):
#     bs = data[:,0]*data[:,3] + data[:,1]*data[:,2]
#     bf = 1/(data[:,0] + data[:,1])
#     return [bs,bf]

# def calculate_bs_bf_nm_nm(data):
#     bs = data[:,3] * data[:,4]
#     bf = 1/(data[:,1] + data[:,3])
#     return [bs,bf]

# def calculate_bs_bf_on_off_nm(data):
#     bs = data[:,3] * data[:,4]
#     bf = 1/(data[:,1] + data[:,3])
#     return [bs,bf]

def calculate_bs_bf(data,model_name="arl"):
    if model_name == "arl":
        bs = data[:,3]*data[:,4]/data[:,5]
        bf = 1/(data[:,0] + data[:,3])
    elif model_name=="on_off_nm":
        bf = 1/(data[:,0] + data[:,1])
        bs = data[:,0]*data[:,2]/data[:,3]
    elif model_name=="afl":
        bf = 1/(data[:,0] + data[:,1])
        bs = data[:,0]*data[:,3] + data[:,1]*data[:,2]
    return [bs,bf]



def plot_image_density(data1,data2,ax,histbins = [101,51],step=2,cmap_limit = [0.1,0.9],cmap_val="Blues", vmax_hist =  0.015,vmin_hist = 0.001,xy_labels = ["Nascent RNA counts","Mature RNA counts"]):
    # vmax_hist =  0.04
    # vmin_hist = 0
    # vmax_hist =  0.015
    # vmin_hist = 0.001
    # step=3
    maxvals = np.array(histbins)-1
    count_hist, _, _ = np.histogram2d(data1,data2,
                bins = [np.arange(histbins[0]+1,step=step)-0.5,np.arange(histbins[1]+1,step=step)-0.5],
                )
    count_hist = count_hist/count_hist.sum()
    count_hist[count_hist<0.001]=0
    original_cmap = plt.get_cmap(cmap_val)
    truncated_cmap = truncate_colormap(original_cmap, cmap_limit[0], cmap_limit[1])
    ax.imshow(count_hist.T,
        origin = 'lower',
        extent = [0.,maxvals[0]+1,0.,maxvals[1]+1],
        vmin = vmin_hist,
        vmax = vmax_hist,
        aspect="auto",
        cmap = truncated_cmap)
    set_font_label(ax,x_label=xy_labels[0],y_label=xy_labels[1])
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)  
def plot_compare_bar(data1,data2,labels,axi_label,ax,x_ticks=[],legend_size=6):
    data2 = data2[data2<=max(data1)]
    density1 = np.histogram(data1, bins=np.arange(min(data1), max(data1)+2), density=True)
    density2 = np.histogram(data2, bins=np.arange(min(data2), max(data2)+2), density=True)

    ax.bar(density1[1][:-1], density1[0], width=0.4, align='center', alpha=0.5, label=labels[0],color=["#C9CACA"])
    ax.bar(density2[1][:-1]+0.4, density2[0], width=0.4, align='center', alpha=0.5, label=labels[1],color=["#009ACE"])
    # ax.set_xlabel(axi_labels[0])
    if len(x_ticks)>0:
        ax.set_xticks(x_ticks)

    set_font_label(ax,x_label=axi_label,y_label="Probability")

    ax.legend( prop={'size':legend_size})


# def plot_hist_density(data1,data2,ax,xy_labels = ["counts","probability"],bw_method=0.15,density_label = 'DynGPT'):
#       
#     ax.hist(data1, bins=int(max(data1))+1, density=True, alpha=0.6, color='b', label='Observed')


#     kde = gaussian_kde(data2,bw_method=bw_method)
#     x_vals = np.linspace(0, max(data1),int(max(data1)))
#     kde_vals = kde(x_vals)
#     kde_vals = kde_vals/sum(kde_vals)
#     
#     ax.plot(x_vals, kde_vals, color='r', label=density_label)
#     x_label,y_label = xy_labels
#     set_font_label(ax,x_label=x_label,y_label=y_label)
#   
#     ax.legend(loc='upper right', prop={'size': 6})