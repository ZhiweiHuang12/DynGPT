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


font_path = '/GPUFS/sysu_jjzhang_1/hzw/data/arial.ttf'
arial_font = fm.FontProperties(fname=font_path)
rcParams['font.family'] = arial_font.get_name()
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['text.usetex'] = False


# Defines a portion of the color map to be intercepted
def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
        cmap(np.linspace(min_val, max_val, n))
    )
    return new_cmap

def plot_hist_2d(ax, data,data_max, label_val,fig,face_color=True,latex_flag=True,cmap_val="Blues",cmap_limit=[0,0.7],xy_lim=[]):
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

def set_font_label(ax,x_label,y_label,font_size=6,font_name='Arial',labelsize=5,tick_width=0.4): # 6.5
    ax.set_xlabel(x_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.set_ylabel(y_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, length=1.6, pad=1,width=tick_width)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize, length=1.6,pad=1,width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return ax

# def set_font_label(ax,x_label,y_label,font_size=7):
#     ax.set_xlabel(x_label, fontsize=font_size,labelpad=1)
#     ax.set_ylabel(y_label, fontsize=font_size,labelpad=1)
#     ax.tick_params(axis='both', which='major', labelsize=6.5, length=1, pad=1)
#     ax.tick_params(axis='both', which='minor', labelsize=6.5, length=1,pad=1)
#     return ax


def set_limit(ax,x_lim,y_lim):
    ax.set_xlim((0,x_lim))
    ax.set_ylim((0,y_lim))

def plot_boxplot(ax,data,species_name,y_ticks=[],latex_flag=False,showfliers=True,palette_color = ['#17becf'],linewidth=0.5,width=0.2,line_color="skyblue",markersize=1.5,line_colors=[]):
    # boxprops=dict(linewidth=0.5)

    boxprops =dict(edgecolor=line_color)
    flierprops=dict(markerfacecolor="#dadad2", marker="o", markersize=markersize,markeredgewidth=0)

    boxplot = sns.boxplot(data=data[species_name], ax=ax,width=width, fliersize=1,palette=palette_color,linewidth=linewidth,showfliers=showfliers,boxprops=boxprops,flierprops=flierprops)

    if len(palette_color)==1:
        plt.setp(boxplot.artists, color=line_color)  
        plt.setp(boxplot.lines, color=line_color)  
    else:
        for artist, color in zip(boxplot.artists, line_colors):
            artist.set_edgecolor(color)
            # artist.set_linewidth(0.5)  
        palette_color_new = [color for color in line_colors for _ in range(6)]

        for line, color in zip(boxplot.lines, palette_color_new):  # Each boxplot has 6 lines（2 whiskers, 2 caps, and 2 medians）
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
def plot_scatter(ax, data1,data2,data_max, species_name,label_vals,stas_type,color = "#000000",y_ticks=[],latex_flag=False,s=2,edgecolor='skyblue'):
    ax.plot([0, data_max], [0, data_max], linestyle='--', color='grey',linewidth=0.5) #'#CF5142'
    if latex_flag:
        label_val = "$"+species_name+"$"
        sns.scatterplot( x=data1[species_name],color=color, y=data2[species_name], label= label_val ,s=s,ax=ax,linewidth=0.1,edgecolor=edgecolor) # s=8
    else:
        sns.scatterplot( x=data1[species_name],color=color, y=data2[species_name], label= species_name ,s=s,ax=ax,linewidth=0.1,edgecolor=edgecolor)# s=8


    legend = ax.legend(loc='upper left', prop={'size': 5},bbox_to_anchor=(0, 0.9, 0.1, 0.1))
    legend.get_frame().set_linewidth(0.5)
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

def plot_scatter_two(ax, data1,data2,label_vals,stas_type,color ="#C3E6F7",y_ticks=[],edgecolor='#1291C1',linewidth1=0.5,linewidth2=0.5,bold_point=[3]): # bold_point=[3,6,7]
    max_val = max(max(data1), max(data2))
    min_val = min(min(data1), min(data2))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='grey',linewidth=linewidth1)
    sns.scatterplot( x=data1,color=color, y=data2,edgecolor=edgecolor,s=8,ax=ax)
    # ax.legend(loc='upper left', prop={'size': 6},bbox_to_anchor=(0, 0.9, 0.1, 0.1))
    if len(bold_point)>0:
        ax.scatter(data1[bold_point],data2[bold_point],s=8,edgecolor='red',facecolor='red', linewidth=linewidth2)
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


def plot_density(ax,values,weights,x_position,x_label,label_value = 'True value',max_param_value=-1):
    kde = gaussian_kde(values, weights=weights)
    x = np.linspace(min(values)*0.8, max(values), 1000)
    y = kde(x)
    # Plotting a weighted KDE curve
    ax.plot(x, y,linewidth=1)
    ax.axvline(x=x_position, color='red', linestyle='--', label=label_value)
    ax.fill_between(x, y, color='blue', alpha=0.05)
    if max_param_value>=0:
        ax.axvline(x=max_param_value, color='green', linestyle='--', label=" max density param")
    set_font_label(ax,x_label,'Posterior prob.')
    ax.legend(prop={'size': 5})

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


def plot_jointplot(data1,data2,xy_ticks,ax,lims,label_names,fontsize = 6,cmap_name = "Blues",fill_flag=True,color="#1FA3DD",data_nn=False,tick_width=0.4,labelsize=5,cmap_limit=[0,1]):#color="#d37166"
    if data_nn:
        label_names = [el + " (DynGPT)" for el in label_names]
    else:
        label_names = [el + " (SSA)" for el in label_names]


    original_cmap = plt.get_cmap(cmap_name)
    truncated_cmap = truncate_colormap(original_cmap, cmap_limit[0], cmap_limit[1])
    # cmap_name = "Blues"
    # g0 = sns.jointplot(x=data1, y=data2, kind="hist",  fill=fill_flag,cmap=cmap_name, color=color,ax=ax,cbar=True)
    marginal_kws={'linewidth': 0.5}
    g0 = sns.jointplot(x=data1, y=data2, kind="kde",  fill=fill_flag,cmap=truncated_cmap, color=color,ax=ax,cbar=True,marginal_kws=marginal_kws)
    if len(lims)>0:
        g0.ax_joint.set_xlim(0, lims[0]+2)
        g0.ax_joint.set_ylim(0, lims[1]+2)
    
    g0.ax_joint.set_xlabel(label_names[0], fontsize=fontsize,labelpad=1)
    g0.ax_joint.set_ylabel(label_names[1], fontsize=fontsize,labelpad=1)

    length = 1.6
    g0.ax_joint.tick_params(axis='both', which='major', labelsize=labelsize, length=length,pad=1,width=tick_width)
    g0.ax_joint.tick_params(axis='both', which='minor', labelsize=labelsize, length=length,pad=1,width=tick_width)
    g0.ax_marg_x.tick_params(axis='x', which='major', labelsize=labelsize, length=length,width=tick_width)  # Set the tick mark length of the upper histogram
    g0.ax_marg_y.tick_params(axis='y', which='major', labelsize=labelsize, length=length,width=tick_width)  # Set the tick mark length of the right histogram
    
    linewidth = 0.5
    for spine in g0.ax_joint.spines.values():
        spine.set_linewidth(linewidth)  # Set the main image border line width to 2
    for spine in g0.ax_marg_x.spines.values():
        spine.set_linewidth(linewidth)  # Set the border line width of the top edge
    for spine in g0.ax_marg_y.spines.values():
        spine.set_linewidth(linewidth)  # Set the border line width on the right side
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
    # noise1 = np.random.uniform(-0.5, 0.5, size=len(data1))
    # noise2 = np.random.uniform(-0.5, 0.5, size=len(data1))
    noise1 = np.random.uniform(0, 1, size=len(data1))
    noise2 = np.random.uniform(0, 1, size=len(data1))

    data1_nn_x_continuous = data1_nn.iloc[:,0].values + noise1
    data1_nn_y_continuous = data1_nn.iloc[:,1].values + noise2

    data1_ssa_x_continuous = data1_ssa.iloc[:,0].values + noise1
    data1_ssa_y_continuous = data1_ssa.iloc[:,1].values + noise2
    x_lim = math.ceil(max(data1_ssa_x_continuous))
    y_lim = math.ceil(max(data1_ssa_y_continuous))
    return data1_nn_x_continuous,data1_nn_y_continuous,data1_ssa_x_continuous,data1_ssa_y_continuous,x_lim,y_lim

def weighted_sample(values,weights,size=5000):
    # values = es_params[param_index][:,param_comp_index]
    # weights = 1/es_loss[param_index] 
    kde = gaussian_kde(values, weights=weights)
    values = np.linspace(min(values)*0.8, max(values), 1000)
    probabilities = kde(values)
    probabilities /= probabilities.sum() 
    sampled_values = np.random.choice(values, size=1500, p=probabilities)
    return sampled_values

def plot_violin(ax,data,true_values,color="#C9CACA",edgecolor="#009ACE", linewidth = 0.5):
    # temp_df = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
   
    sns.violinplot(data=data, ax=ax,inner=None,color=color,edgecolor=edgecolor,linewidth=0.1)
    ax.tick_params(axis='both', which='major', labelsize=6)
    line_length = len(true_values)
    # line_length = 3
    for j, true_value in enumerate(true_values):
        if j==0:
            ax.axhline(y=true_value, xmin=j/line_length, xmax=(j+1)/line_length, color='grey', linestyle='--',label='True value',linewidth=linewidth)
        else:
            ax.axhline(y=true_value, xmin=j/line_length, xmax=(j+1)/line_length, color='grey', linestyle='--',linewidth=linewidth)

    set_font_label(ax, "Parameters", "Value")
    ax.legend(loc='upper left', prop={'size': 6},bbox_to_anchor=(0, 0.9, 0.1, 0.1))


def plot_hist_density(data1,data2,ax,xy_labels = ["Counts","Probability"],bw_method=0.15,density_label = 'DynGPT',hist_color = "b"):
    ax.hist(data1, bins=int(max(data1))+1, density=True, alpha=0.6, color=hist_color, label='Data')
    kde = gaussian_kde(data2,bw_method=0.15)
    x_vals = np.linspace(0, max(data1),int(max(data1)))
    kde_vals = kde(x_vals)
    kde_vals = kde_vals/sum(kde_vals)
    ax.plot(x_vals, kde_vals, color='r',linewidth=1, label=density_label)
    x_label,y_label = xy_labels
    set_font_label(ax,x_label=x_label,y_label=y_label)
    ax.legend(loc='upper right', prop={'size': 5})

# def plot_hist_density(data1,data2,ax,xy_labels = ["counts","probability"],bw_method=0.15,density_label = 'DynGPT'):
#        
#     ax.hist(data1, bins=int(max(data1))+1, density=True, alpha=0.6, color='b', label='Observed')

#    
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