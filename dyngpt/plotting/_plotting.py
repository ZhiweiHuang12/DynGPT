import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoLocator
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np
import sys 
import os
from collections import Counter
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from scipy.special import rel_entr
import itertools
from scipy.stats import gaussian_kde
# print(os.getcwd())
from dyngpt._utils._util_plotting import *
import math
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from sklearn.neighbors import KernelDensity

label_values = ["Observed","DynGPT"]
def plot_afl_inference_data_burst_distribution(observed_datas, nn_sample_datas, es_params, es_loss,es_param, figure_dir="", model_name="afl",synthetic=0,show_flag=False):
    """
    Plot the burst distribution of AFL inference data, comparing observed data with neural network samples.

    Parameters:
    observed_datas (ndarray): Array containing observed data for various parameters.
    nn_sample_datas (ndarray): Array containing neural network sample data corresponding to observed data.
    es_params (ndarray): Array of estimated parameters from the AFL model.
    es_loss (ndarray): Array of loss values associated with the estimated parameters.
    figure_dir (str): Directory path where the plots will be saved.
    model_name (str): Name of the model, used in the filenames of the saved plots. Default is "afl".

    """
    # Labels for the x-axis in the plots
    x_labels = ["Burst size", "Burst frequency"]
    es_bs_bf = calculate_bs_bf(np.exp(es_param),model_name=model_name)
    # Index for observed data
    observed_data_index = [1]

    # Define plot dimensions in inches based on width in millimeters
    width_mm = 183
    width_inch = width_mm * 0.0393701
    height_inch = width_inch * 0.21

    # Range of parameter indices to loop over
    param_index_li = range(0, observed_datas.shape[0])

    for param_index in param_index_li:
        # Clear the current figure
        plt.clf()
        # Create a new figure with specified size
        fig = plt.figure(figsize=(width_inch, height_inch))

        # Define the grid layout with 1 row and 4 columns
        gs = GridSpec(1, 4, width_ratios=[0.75, 1, 1, 1])

        # Scatter plot for mean comparison between observed and neural network data
        ax = fig.add_subplot(gs[0])
        plot_scatter_two(ax, np.mean(observed_datas, axis=1), np.mean(nn_sample_datas[:, :, 1], axis=1), label_values, "Mean")

        # Histogram and density plot for the first parameter
        ax = fig.add_subplot(gs[1])
        plot_hist_density(observed_datas[param_index, :], nn_sample_datas[param_index, :, observed_data_index[0]], ax, bw_method=0.15, hist_color="#CBCED3",xy_labels = ["mRNA counts","Probability"])

        # Loop over the burst size and frequency components
        for param_comp_index in range(2):
            ax = fig.add_subplot(gs[param_comp_index + 2])

            # Calculate burst size/frequency from exponential parameters
            temp_data = np.exp(es_params[param_index])
            values = calculate_bs_bf(temp_data,model_name=model_name)[param_comp_index]

            # Weighting based on loss values
            weights = 1 / es_loss[param_index]

            # Density plot for estimated values
            plot_density(ax, values, weights, es_bs_bf[param_comp_index][param_index], x_labels[param_comp_index], label_value="Estimated value")

        # Adjust layout and save the plots as JPG and EPS
        plt.tight_layout()
        if len(figure_dir)>0:
            plt.savefig(figure_dir + "{}_posterior_{}_{}_observed.jpg".format(model_name, param_index,synthetic), dpi=300)
            plt.savefig(figure_dir + "{}_posterior_{}_{}_observed.eps".format(model_name, param_index,synthetic), dpi=300)
        if show_flag:
            plt.show()
        else:
            plt.close()

def plot_nm_nm_inference_data_burst_distribution(observed_datas, nn_sample_datas, es_params, es_loss,es_param, figure_dir, model_name="nm_nm",observed_data_index = [1,2],vline_flag=False):

    histbins_li = [[100,40],[100,40],[40,60],[10,10],[10,10]]
    xy_lim = [100,50]
    histbins = histbins_li[0]
    cmap_limit = [0,1]
    step = 3
    x_labels = ["Burst size", "Burst frequency"]
    # es_bs_bf = calculate_bs_bf_nm_nm(np.exp(es_param))
    es_bs_bf = calculate_bs_bf(np.exp(es_param),model_name=model_name)

    # plot inference density, distribution and mean
    width_mm = 183
    width_inch = width_mm * 0.0393701
    height_inch = width_inch * 0.38 
    
    param_index_li = range(0,observed_datas.shape[0])
    # for param_index in range(1,100):

    for param_index in param_index_li:
        plt.clf()
        fig = plt.figure(figsize=(width_inch, height_inch))
        gs = GridSpec(2, 4, width_ratios=[0.8,1,1.2,1.2])
        i=param_index

        ax = fig.add_subplot(gs[0,0])
        observed_nrna_mean = np.mean(observed_datas[:,:,0],axis=1)
        sampled_nrna_mean = np.mean(nn_sample_datas[:,:,1],axis=1)
        plot_scatter_two(ax,observed_nrna_mean,sampled_nrna_mean,label_values,"Mean")

        ax = fig.add_subplot(gs[1,0])
        observed_mrna_mean = np.mean(observed_datas[:,:,1],axis=1)
        sampled_mrna_mean = np.mean(nn_sample_datas[:,:,2],axis=1)
        plot_scatter_two(ax,observed_mrna_mean,sampled_mrna_mean,label_values,"Mean")

        ax0 = fig.add_subplot(gs[0,1])
        ax1 = fig.add_subplot(gs[1,1])

        histbins = observed_datas[i].astype("int").max(axis=0)+1
        plot_image_density(observed_datas[i][:,0],observed_datas[i][:,1],ax0,cmap_limit = cmap_limit,step = step,histbins=histbins)
        plot_image_density(nn_sample_datas[i][:,observed_data_index][:,0],nn_sample_datas[i][:,observed_data_index][:,1],ax1,cmap_limit = cmap_limit,step = step,histbins=histbins)

        ax2 = fig.add_subplot(gs[0,2])
        ax3 = fig.add_subplot(gs[1,2])
        bw_method = 0.3
        plot_hist_density(observed_datas[i][:,0],nn_sample_datas[i,:,observed_data_index[0]],ax2,bw_method=0.4,hist_color="#CBCED3")
        plot_hist_density(observed_datas[i][:,1],nn_sample_datas[i,:,observed_data_index[1]],ax3,bw_method=bw_method,hist_color="#CBCED3")
        if model_name == "nm_nm":
            for param_comp_index in range(2):
                ax = fig.add_subplot(gs[param_comp_index,3])
                temp_data = np.exp(es_params[param_index])
                values = calculate_bs_bf_nm_nm(temp_data)[param_comp_index]
                values = calculate_bs_bf_nm_nm(temp_data,model_name=model_name)[param_comp_index]

                weights = 1/es_loss[param_index]  # 对应于每个值的权重
                plot_density(ax,values,weights, es_bs_bf[param_comp_index][param_index],x_labels[param_comp_index],vline_flag=vline_flag)
            
        plt.tight_layout()
        plt.savefig(figure_dir + "{}_density_mean_{}_observed.jpg".format(model_name,param_index),dpi=300)
        plt.savefig(figure_dir + "{}_density_mean_{}_observed.eps".format(model_name,param_index),dpi=300)
        plt.close()


def plot_isc_compare_distribution(observed_datas, nn_sample_datas,figure_dir,model_name):
    cols_name = ["X_"+str(i) for i in range(observed_datas.shape[2])]
    species_index = np.array([1,2,4,5])
    for k in range(observed_datas.shape[0]):
        observed_data = observed_datas[k]
        nn_sample_data = nn_sample_datas[k]
        observed_data = pd.DataFrame(observed_data,columns=cols_name)
        nn_sample_data = pd.DataFrame(nn_sample_data,columns=cols_name)

        width_mm = 183
        width_inch = width_mm * 0.0393701
        height_inch = width_inch*0.8 
        
        fig, axes = plt.subplots(len(species_index), len(species_index), figsize=(width_inch, height_inch))
        labels = label_values
        x_ticks = [[],[],[],[]]
        cmap_limit = [0.4,0.5]
        cmap_limit = [0.1,0.8]

        for i in range(len(species_index)):
            for j in range(len(species_index)):
                if i==j:
                    axi_label = "${}$ counts".format(cols_name[species_index[i]])
                    plot_compare_bar(observed_data.iloc[:,species_index[i]],nn_sample_data.iloc[:,species_index[i]],labels,axi_label,axes[i, j],x_ticks[i],legend_size=5)
                elif i<j:
                    data_sub = observed_data.iloc[:,species_index[[i,j]]]
                    data_max = data_sub.max()
                    plot_hist_2d(axes[i, j],data_sub,data_max,"SSA",fig,False,latex_flag= True,cmap_limit=cmap_limit,xy_lim=[20,18],vmax=0.2)
                elif i>j:
                    data_sub = nn_sample_data.iloc[:,species_index[[j,i]]]
                    data_max = observed_data.iloc[:,species_index[[j,i]]].max()
                    plot_hist_2d(axes[i, j],data_sub,data_max,label_values[1],fig,False,latex_flag= True,cmap_limit=cmap_limit,xy_lim=[20,18],vmax=0.2)
        plt.tight_layout()
        plt.savefig(figure_dir + "{}_joint_prob_{}.jpg".format(model_name,k),dpi=400)
        plt.savefig(figure_dir + "{}_joint_prob_{}.pdf".format(model_name,k))
        plt.savefig(figure_dir + "{}_joint_prob_{}.eps".format(model_name,k))


def plot_hist_2d(ax, data,data_max, label_val,fig,face_color=True,latex_flag=True,cmap_val="Blues",cmap_limit=[0,0.7],xy_lim=[],vmax=0.025,vmin=0.025,font_size=6,label_size=5):
    """Plot a 2D histogram on the given axes with custom color mapping and labels.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the histogram.
            data (pandas.DataFrame): The data containing the values to be plotted.
            data_max (list): Maximum values for the data along the x and y axes.
            label_val (str): Label for the axis to be displayed.
            fig (matplotlib.figure.Figure): The figure object that holds the axes.
            face_color (bool, optional): Whether to show a color bar. Defaults to True.
            latex_flag (bool, optional): Whether to format labels in LaTeX. Defaults to True.
            cmap_val (str, optional): The colormap to use for the plot. Defaults to "Blues".
            cmap_limit (list, optional): The lower and upper limits for the colormap. Defaults to [0, 0.7].
            xy_lim (list, optional): The limits for the x and y axes. Defaults to an empty list, which uses data_max.
            vmax (float, optional): The maximum value for the color scale. Defaults to 0.025.
            vmin (float, optional): The minimum value for the color scale. Defaults to 0.025.
            font_size (int, optional): The font size for labels. Defaults to 6.
            label_size (int, optional): The font size for tick labels. Defaults to 5.

    """    
    col_names = data.columns.tolist()
    if len(xy_lim)>0:
        data = data[data[col_names[0]]<= xy_lim[0]]
        data = data[data[col_names[1]]<= xy_lim[1]]
    else:
        data = data[data[col_names[0]]<= data_max[0]]
        data = data[data[col_names[1]]<= data_max[1]]
    data_max = data.max().astype("int")
    original_cmap = plt.get_cmap(cmap_val)
    truncated_cmap = truncate_colormap(original_cmap, cmap_limit[0], cmap_limit[1])
    hist = ax.hist2d(data[col_names[0]], data[col_names[1]], bins=(data_max.values), cmap=truncated_cmap,density=True,norm=mpl.colors.LogNorm(vmax=0.025),)
    
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
    set_font_label(ax,x_label=x_label,y_label=y_label,font_size=font_size,labelsize=label_size)
    ax.set_facecolor('white')
    if face_color:
        cbar1 = fig.colorbar(hist[3], ax=ax)
        cbar1.ax.tick_params(labelsize=6)


def set_font_label(ax,x_label,y_label,font_size=6,font_name='Arial',labelsize=5,tick_width=0.4): # 6.5
    """Set font styles and sizes for axis labels and tick labels.

    Args:
        ax (matplotlib.axes.Axes): The axes for which to set the labels.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        font_size (int, optional): Font size for axis labels. Defaults to 6.
        font_name (str, optional): Font name for the labels. Defaults to 'Arial'.
        labelsize (int, optional): Font size for tick labels. Defaults to 5.
        tick_width (float, optional): Width of the tick marks. Defaults to 0.4.

    Returns:
        ax (matplotlib.axes.Axes): The axes with the updated label settings.

    """  
    ax.set_xlabel(x_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.set_ylabel(y_label, fontsize=font_size, fontname=font_name,labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, length=1.6, pad=1,width=tick_width)
    ax.tick_params(axis='both', which='minor', labelsize=labelsize, length=1.6,pad=1,width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return ax


def set_limit(ax,x_lim,y_lim):
    """Set the x and y axis limits for the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes for which to set the limits.
        x_lim (float): The maximum value for the x-axis.
        y_lim (float): The maximum value for the y-axis.

    """
    ax.set_xlim((0,x_lim))
    ax.set_ylim((0,y_lim))


def plot_boxplot(ax,data,species_name,y_ticks=[],latex_flag=False,showfliers=True,palette_color = ['#17becf'],linewidth=0.5,width=0.2,line_color="skyblue",markersize=1.5,line_colors=[],font_size=6,labelsize=5,legend_size=5):
    """Plot a boxplot with customization options for states and formatting.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot the boxplot.
        data (pandas.DataFrame): The data containing the species information.
        species_name (str or list): The name or list of species to plot.
        y_ticks (list, optional): Specific y-ticks to display. Defaults to an empty list.
        latex_flag (bool, optional): Whether to format labels in LaTeX. Defaults to False.
        showfliers (bool, optional): Whether to show outliers in the boxplot. Defaults to True.
        palette_color (list, optional): List of colors for the boxplot. Defaults to ['#17becf'].
        linewidth (float, optional): Line width for the boxplot. Defaults to 0.5.
        width (float, optional): Width of the boxplot boxes. Defaults to 0.2.
        line_color (str, optional): Color for the boxplot lines. Defaults to "skyblue".
        markersize (float, optional): Size of the markers for outliers. Defaults to 1.5.
        line_colors (list, optional): List of line colors for the boxplot. Defaults to an empty list.
        font_size (int, optional): Font size for axis labels. Defaults to 6.
        labelsize (int, optional): Font size for tick labels. Defaults to 5.
        legend_size (int, optional): Font size for legend. Defaults to 5.

    """    
    boxprops =dict(edgecolor=line_color)
    flierprops=dict(markerfacecolor="#dadad2", marker="o", markersize=markersize,markeredgewidth=0)
    boxplot = sns.boxplot(data=data[species_name], ax=ax,width=width, fliersize=1,palette=palette_color,linewidth=linewidth,showfliers=showfliers,boxprops=boxprops,flierprops=flierprops)

    if len(palette_color)==1:
        plt.setp(boxplot.artists, color=line_color)  
        plt.setp(boxplot.lines, color=line_color)  
    else:
        for artist, color in zip(boxplot.artists, line_colors):
            artist.set_edgecolor(color)
        palette_color_new = [color for color in line_colors for _ in range(6)]

        for line, color in zip(boxplot.lines, palette_color_new):  # Each boxplot has 6 lines（2 whiskers, 2 caps, and 2 medians）
            line.set_color(color)

    if latex_flag:
        labels = ["${}$".format(el) for el in species_name]
        ax.set_xticks(ticks=range(len(species_name)), labels=labels)


    if isinstance(species_name, list):
        set_font_label(ax,x_label="Species",y_label="KL distance",font_size=font_size,labelsize=labelsize)
    else:
        set_font_label(ax,x_label="Species",y_label="KL distance",font_size=font_size,labelsize=labelsize)

    if len(y_ticks)>0:
        ax.set_yticks(y_ticks)
    

def plot_scatter(ax, data1,data2,data_max, species_name,label_vals,stas_type,color = "#000000",y_ticks=[],latex_flag=False,s=2,edgecolor='skyblue',font_size=6,labelsize=5,legend_size=5):
    """
    Plot a scatter plot of two datasets on the given axes, with the option to customize appearance and labels.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the scatter plot on.
        data1 (pandas.DataFrame): The first dataset to be plotted.
        data2 (pandas.DataFrame): The second dataset to be plotted.
        data_max (float): The maximum value used to set the axis limits.
        species_name (str): The name of the species to be used for labeling.
        label_vals (list): The values used for labeling the axes.
        stas_type (str): The type of statistics to display on the axes labels.
        color (str, optional): The color of the points in the scatter plot. Defaults to "#000000".
        y_ticks (list, optional): The tick values for the y-axis. Defaults to [].
        latex_flag (bool, optional): Whether to display the species name as LaTeX-formatted. Defaults to False.
        s (int, optional): The size of the scatter plot points. Defaults to 2.
        edgecolor (str, optional): The edge color of the scatter plot points. Defaults to 'skyblue'.
        font_size (int, optional): The font size for axis labels. Defaults to 6.
        labelsize (int, optional): The font size for labels. Defaults to 5.
        legend_size (int, optional): The font size for the legend. Defaults to 5.
    """ 
    ax.plot([0, data_max], [0, data_max], linestyle='--', color='grey',linewidth=0.5) #'#CF5142'
    if latex_flag:
        label_val = "$"+species_name+"$"
        sns.scatterplot( x=data1[species_name],color=color, y=data2[species_name], label= label_val ,s=s,ax=ax,linewidth=0.1,edgecolor=edgecolor) # s=8
    else:
        sns.scatterplot( x=data1[species_name],color=color, y=data2[species_name], label= species_name ,s=s,ax=ax,linewidth=0.1,edgecolor=edgecolor)# s=8
    legend = ax.legend(loc='upper left', prop={'size': legend_size},bbox_to_anchor=(0, 0.9, 0.1, 0.1))
    legend.get_frame().set_linewidth(0.5)
    x_label = "{} ({})".format(stas_type,label_vals[0])
    y_label = "{} ({})".format(stas_type,label_vals[1])
    ax.set_xlim(0, data_max*1.05)  
    ax.set_ylim(0, data_max*1.05) 
    if len(y_ticks)>0:
        ax.set_yticks(y_ticks)
        ax.set_xticks(y_ticks)

    set_font_label(ax,x_label=x_label,y_label=y_label,font_size=font_size,labelsize=labelsize)
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False) 

def plot_scatter_two(ax, data1,data2,label_vals,stas_type,color ="#C3E6F7",y_ticks=[],edgecolor='#1291C1',linewidth1=0.5,linewidth2=0.5,bold_point=[3]): # bold_point=[3,6,7]
    """
    Plot a scatter plot of two datasets (data1 and data2) with a reference line and optional highlighting of specific points.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the scatter plot on.
        data1 (pandas.DataFrame): The first dataset to be plotted.
        data2 (pandas.DataFrame): The second dataset to be plotted.
        label_vals (list): The values used for labeling the axes.
        stas_type (str): The type of statistics to display on the axes labels.
        color (str, optional): The color of the points in the scatter plot. Defaults to "#C3E6F7".
        y_ticks (list, optional): The tick values for the y-axis. Defaults to [].
        edgecolor (str, optional): The edge color of the scatter plot points. Defaults to '#1291C1'.
        linewidth1 (float, optional): The width of the reference line. Defaults to 0.5.
        linewidth2 (float, optional): The width of the highlighted points. Defaults to 0.5.
        bold_point (list, optional): Indices of points to be highlighted. Defaults to [3].
    """ 
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

def plot_density(ax,values,weights,x_position,x_label,label_value = 'Estimated value',vline_flag = True,mean_max_flag=True,max_density_val=2,linewidth=0.75, color='#E61F19',v_color='#E184FF',fill_color="#EBAC80"):
    """
    Plot a density estimate (using kernel density estimation).

    Args:
        ax (matplotlib.axes.Axes): The axes to plot the density plot on.
        values (array-like): The values for which the density is estimated.
        weights (array-like): The weights associated with the values.
        x_position (float): The x position for a vertical line indicating the density estimate.
        x_label (str): The label for the x-axis.
        label_value (str, optional): The label for the density estimate line. Defaults to 'Estimated value'.
        vline_flag (bool, optional): Whether to display a vertical line at the estimated value. Defaults to True.
        mean_max_flag (bool, optional): Whether to include mean and max vertical lines. Defaults to True.
        max_density_val (int, optional): The maximum density value to display. Defaults to 2.
        linewidth (float, optional): The width of the lines. Defaults to 0.75.
        color (str, optional): The color of the density plot line. Defaults to '#E61F19'.
        v_color (str, optional): The color of the vertical lines. Defaults to '#E184FF'.
        fill_color (str, optional): The color to fill the area under the density curve. Defaults to "#EBAC80".

    Returns:
        (numpy.ndarray, numpy.ndarray): The x and y values of the density estimate.
    """ 
    kde = gaussian_kde(values, weights=weights)
    x = np.linspace(min(values)*0.8, max(values), 1000)
    y = kde(x)
    ax.plot(x, y,linewidth=linewidth, color=color)
    ax.fill_between(x, y, color=fill_color,alpha=0.05)
    if vline_flag:
        max_density_val = get_max_density_point(values,weights)
        ax.axvline(x=x_position, color='red', linestyle='--', label="Max density",linewidth=linewidth)
        ax.legend(prop={'size': 5})
    set_font_label(ax,x_label,'Posterior prob.')
    ax.spines['top'].set_visible(False)  
    ax.spines['right'].set_visible(False)
    return x,y

def plot_jointplot(data1,data2,xy_ticks,ax,lims,label_names,fontsize = 6,cmap_name = "Blues",fill_flag=True,color="#d37166",data_nn=False):
    """
    Plot a jointplot of two variables with kernel density estimation. Customizes axis ticks and labels.

    Args:
        data1 (array-like): The first data set for the x-axis.
        data2 (array-like): The second data set for the y-axis.
        xy_ticks (list of lists): Custom tick values for the x and y axes.
        ax (matplotlib.axes.Axes): The axes on which to plot the jointplot.
        lims (list): Axis limits for the plot.
        label_names (list): List of labels for the x and y axes.
        fontsize (int, optional): Font size for the labels, defaults to 6.
        cmap_name (str, optional): Colormap name for KDE, defaults to "Blues".
        fill_flag (bool, optional): Whether to fill the density plot, defaults to True.
        color (str, optional): Color for the KDE plot, defaults to "#d37166".
        data_nn (bool, optional): Whether to use a neural network-derived label, defaults to False.

    Returns:
        seaborn.JointGrid: The jointplot object.
    """   
    if data_nn:
        label_names = [el + " (DynGPT)" for el in label_names]
    else:
        label_names = [el + " (SSA)" for el in label_names]
    cmap_name = "Blues"
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
    return g0

def plot_violin(ax,data,true_values,color="#C9CACA",edgecolor="skyblue",legend_flag=False,legend_size=6,font_size=6,labelsize=5,log_flag = True):
    """
    Plot a violin plot with overlaid horizontal lines indicating the true values. Optionally includes a legend.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot the violin plot.
        data (array-like): The data to plot in the violin plot.
        true_values (array-like): The true values to be highlighted with horizontal lines.
        color (str, optional): The fill color for the violins, defaults to "#C9CACA".
        edgecolor (str, optional): The edge color for the violins, defaults to "skyblue".
        legend_flag (bool, optional): Whether to display the legend, defaults to False.
        legend_size (int, optional): Font size for the legend, defaults to 6.
        font_size (int, optional): Font size for the axis labels, defaults to 6.
        labelsize (int, optional): Font size for the tick labels, defaults to 5.
        log_flag (bool, optional): Whether to log-transform the y-axis, defaults to True.

    Returns:
        None
    """  
    sns.violinplot(data=data, ax=ax,inner=None,color=color,edgecolor=edgecolor)
    ax.tick_params(axis='both', which='major', labelsize=6)
    for j, true_value in enumerate(true_values):
        if j==0:
            ax.axhline(y=true_value, xmin=j/len(true_values), xmax=(j+1)/len(true_values), color='grey', linestyle='--',label='True value')
        else:
            ax.axhline(y=true_value, xmin=j/len(true_values), xmax=(j+1)/len(true_values), color='grey', linestyle='--')

    if log_flag:
        y_label = r"$\log_2 \, \mathrm{Value}$"
    else:
        y_label = "Value"
    set_font_label(ax, "Parameters",y_label,font_size=font_size,labelsize=labelsize)
    if legend_flag:
        ax.legend(loc='upper left', prop={'size': legend_size},bbox_to_anchor=(0, 0.9, 0.1, 0.1))


def plot_hist_density(data1,data2,ax,xy_labels = ["Counts","Probability"],bw_method=0.15,density_label = 'DynGPT',hist_color = "b",linewidth=0.75,color="#E61F19"):
    """
    Plot a histogram of data1 and overlays a kernel density estimate for data2. The KDE is normalized to the probability density.

    Args:
        data1 (array-like): The first data set for the histogram.
        data2 (array-like): The second data set for the KDE.
        ax (matplotlib.axes.Axes): The axes on which to plot the histogram and KDE.
        xy_labels (list, optional): Labels for the x and y axes, defaults to ["Counts", "Probability"].
        bw_method (float, optional): The bandwidth parameter for the KDE, defaults to 0.15.
        density_label (str, optional): The label for the density curve, defaults to 'DynGPT'.
        hist_color (str, optional): Color for the histogram, defaults to "b".
        linewidth (float, optional): Line width for the density plot, defaults to 0.75.
        color (str, optional): Color for the density curve, defaults to "#E61F19".

    Returns:
        None
    """   
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


def plot_image_density(data1,data2,ax,histbins = [101,51],step=2,cmap_limit = [0.1,0.9],cmap_val="Blues", vmax_hist =  0.015,vmin_hist = 0.001,xy_labels = ["Nascent RNA counts","Mature RNA counts"]):
    """
    Plot a 2D density heatmap comparing two datasets.

    Args:
        data1 (array-like): The first dataset (e.g., Nascent RNA counts).
        data2 (array-like): The second dataset (e.g., Mature RNA counts).
        ax (matplotlib.axes.Axes): The axis object where the plot will be drawn.
        histbins (list, optional): The number of bins for the histogram along each axis. Defaults to [101, 51].
        step (int, optional): The step size for binning. Defaults to 2.
        cmap_limit (list, optional): The lower and upper limits for the colormap. Defaults to [0.1, 0.9].
        cmap_val (str, optional): The name of the colormap to use. Defaults to "Blues".
        vmax_hist (float, optional): The maximum value for color scaling. Defaults to 0.015.
        vmin_hist (float, optional): The minimum value for color scaling. Defaults to 0.001.
        xy_labels (list, optional): Labels for the x and y axes. Defaults to ["Nascent RNA counts", "Mature RNA counts"].
    """   
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


def plot_compare_bar(data1,data2,labels,axi_label,ax,x_ticks=[],legend_size=6,font_size=6,labelsize=5):
    """
    Plot a comparison bar chart between two datasets.

    Args:
        data1 (array-like): The first dataset.
        data2 (array-like): The second dataset.
        labels (list): The labels for the two datasets.
        axi_label (str): The label for the x-axis.
        ax (matplotlib.axes.Axes): The axis object where the plot will be drawn.
        x_ticks (list, optional): List of custom tick positions for the x-axis. Defaults to [].
        legend_size (int, optional): Font size for the legend. Defaults to 6.
        font_size (int, optional): Font size for the plot. Defaults to 6.
        labelsize (int, optional): Font size for the axis labels. Defaults to 5.
    """ 
    data2 = data2[data2<=max(data1)]
    density1 = np.histogram(data1, bins=np.arange(min(data1), max(data1)+2), density=True)
    density2 = np.histogram(data2, bins=np.arange(min(data2), max(data2)+2), density=True)

    ax.bar(density1[1][:-1], density1[0], width=0.4, align='center', alpha=0.5, label=labels[0],color=["#C9CACA"])
    ax.bar(density2[1][:-1]+0.4, density2[0], width=0.4, align='center', alpha=0.5, label=labels[1],color=["#009ACE"])
    # ax.set_xlabel(axi_labels[0])
    if len(x_ticks)>0:
        ax.set_xticks(x_ticks)

    set_font_label(ax,x_label=axi_label,y_label="Probability",font_size=font_size,labelsize=labelsize)

    ax.legend( prop={'size':legend_size})

def plot_loss(losses):
    """
    Plot the loss values over epochs.

    Args:
        losses (array-like): The loss values to plot, typically from a training process.
    """   
    plt.figure(figsize=(3, 1.6))  # Set canvas size
    plt.plot(losses, color='royalblue')  # Research color scheme
    plt.xlabel('Epoch', fontsize=8)
    plt.ylabel('Loss value', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

def plot_model_comparison_stats(result_valid_set,state_index=[],label_values = ["Simulation","DynGPT"]):
    """
    Plot a comparison of statistical properties (mean, standard deviation, and Kullback-Leibler divergence) 
    between simulation and DynGPT results for a given set of data.

    Args:
        result_valid_set (dict): Dictionary containing the simulation results, including 'mean_val_ssa', 
                                  'mean_val_nn', 'std_val_ssa', 'std_val_nn', and 'kl_div'.
        state_index (list, optional): List of indices for the states to plot. Defaults to an empty list, 
                                      which uses all available states.
        label_values (list, optional): Labels to use for the legend, typically ["Simulation", "DynGPT"]. 
                                       Defaults to ["Simulation", "DynGPT"].
    """
    if len(state_index)==0:
        state_index = list(range(np.array(result_valid_set["ssa_sample"]).shape[2]))
#     s_i,s_j =state_index
    species_names = result_valid_set["kl_div"].columns.tolist()
    mean_val_ssa,mean_val_nn = result_valid_set['mean_val_ssa'], result_valid_set['mean_val_nn']
    std_val_ssa,std_val_nn = result_valid_set['std_val_ssa'], result_valid_set['std_val_nn']
    kl_val = result_valid_set['kl_div']
    scatter_colors = ["#f9e1d1", "#c6dae9", "#ceead6","#f3cece"]
    edgecolors = ["#df592a","#489bc5","#32a852","#d22427"]
    rows = math.ceil(len(state_index)/2)
    width_mm=183*1.3
    width_inch = width_mm * 0.0393701
    height_inch = width_inch*0.5 * 0.35*rows
    fig = plt.figure(figsize=(width_inch, height_inch),constrained_layout=True)
    
    gs = GridSpec(rows, 6, width_ratios=[1, 1, 0.4,1, 1, 0.4])
    y_ticks_li = [[] for i in range(3*len(state_index))]

    for i in range(len(state_index)):
        s_i = state_index[i]
        mean_nn_max,mean_ssa_max = mean_val_nn.max(axis=0),mean_val_ssa.max(axis=0)
        std_nn_max,std_ssa_max = std_val_nn.max(axis=0),std_val_ssa.max(axis=0)
        plot_scatter(fig.add_subplot(gs[0+i*3]),mean_val_nn,mean_val_ssa,max(mean_nn_max[s_i],mean_ssa_max[s_i]),species_names[s_i],label_values,"Mean",y_ticks=y_ticks_li[0],latex_flag=False,color=scatter_colors[i],edgecolor=edgecolors[i],font_size=8,labelsize=8,legend_size=8,s=6)
        plot_scatter(fig.add_subplot(gs[1+i*3]),std_val_nn,std_val_ssa,max(std_nn_max[s_i],std_ssa_max[s_i]),species_names[s_i],label_values,"SD",y_ticks=y_ticks_li[1],latex_flag=False,color=scatter_colors[i],edgecolor=edgecolors[i],font_size=8,labelsize=8,legend_size=8,s=6)
        plot_boxplot(fig.add_subplot(gs[2+i*3]),kl_val,[species_names[s_i]],y_ticks=y_ticks_li[2],latex_flag=False,palette_color = [scatter_colors[i]],line_color=edgecolors[i],font_size=8,labelsize=8,legend_size=8,showfliers=False)
    plt.tight_layout()

def plot_param_posterior_dist(result_infer,excluded_indexes=[],param_indexes=[0,1],true_values = []):
    """
    Plot the posterior distributions of inferred parameters, along with optional true values for comparison.

    Args:
        result_infer (dict): Dictionary containing inferred parameters ('es_params'), parameter names ('param_names'), 
                             and loss values ('es_loss').
        excluded_indexes (list, optional): List of indexes to exclude from the parameter samples. Defaults to an empty list.
        param_indexes (list, optional): List of indexes for the parameters to plot. Defaults to [0, 1].
        true_values (list, optional): List of true values for the parameters to compare against. Defaults to an empty list.
    """      
    es_params = result_infer["es_params"]
    data_col_names = result_infer["param_names"]
    es_loss = result_infer["es_loss"]
    scatter_color = "#C1E8FB"  #009ACE
    scatter_color = "#009ACE"  #009ACE

    width_mm = 183
    width_inch = width_mm * 0.0393701
    height_inch = width_inch*0.3
    fig, axes = plt.subplots(nrows=1, ncols=len(param_indexes), figsize=(width_inch, height_inch))
    for i in range(len(param_indexes)):

        param_index = param_indexes[i]
        if len(true_values)>0:
            true_value = true_values[param_index]
        else:
            true_value = []
        sampled_values_li = [weighted_sample(es_params[param_index][:,param_comp_index],1/es_loss[param_index])  for param_comp_index in range(len(data_col_names))]
        
        data = pd.DataFrame(np.array(sampled_values_li).T,columns=data_col_names)
        # Create grid of subplots
        plot_violin(axes[i],data,true_value,edgecolor = scatter_color,color="#DAEDF4",legend_size=8,font_size=8,labelsize=8)
        plt.tight_layout()

def plot_distribution_comparison_nd(result_infer,species_index=[0,1,2],data_index = [1]):
    """
    Plot a comparison of observed and predicted distributions, using 2D histograms and bar plots to visualize the differences between data and model predictions.
    
    Args:
        result_infer (dict): Dictionary containing observed data ('observed_datas') and model predictions ('nn_sample_datas').
        species_index (list, optional): List of indices for the species to compare. Defaults to [0, 1, 2].
        data_index (list, optional): List of indexes specifying which data sets to compare. Defaults to [1].
    """
    labels = ["Data","DynGPT"]
    label_values = ["DynGPT","Data"]
    cols_name = result_infer["state_name"]
    species_index = np.array(species_index)
    observed_datas, nn_sample_datas = result_infer['observed_datas'],result_infer['nn_sample_datas']
    
    for k in data_index:
        observed_data = observed_datas[k]
        nn_sample_data = nn_sample_datas[k]
        observed_data = pd.DataFrame(observed_data,columns=cols_name)
        nn_sample_data = pd.DataFrame(nn_sample_data,columns=cols_name)

        width_mm = 183
        width_inch = width_mm * 0.0393701
        height_inch = width_inch*0.8 
        
        fig, axes = plt.subplots(len(species_index), len(species_index), figsize=(width_inch, height_inch))
        x_ticks = [[],[],[],[]]
        cmap_limit = [0.4,0.5]
        cmap_limit = [0.1,0.8]

        for i in range(len(species_index)):
            for j in range(len(species_index)):
                if i==j:
                    axi_label = "${}$ counts".format(cols_name[species_index[i]])
                    plot_compare_bar(observed_data.iloc[:,species_index[i]],nn_sample_data.iloc[:,species_index[i]],labels,axi_label,axes[i, j],x_ticks[i],legend_size=8,font_size=8,labelsize=8)
                elif i<j:
                    data_sub = observed_data.iloc[:,species_index[[i,j]]]
                    data_max = data_sub.max()
                    plot_hist_2d(axes[i, j],data_sub,data_max,label_values[0],fig,False,latex_flag= True,cmap_limit=cmap_limit,xy_lim=[],vmax=0.5,font_size=8,label_size=8)
                elif i>j:
                    data_sub = nn_sample_data.iloc[:,species_index[[j,i]]]
                    data_max = observed_data.iloc[:,species_index[[j,i]]].max()
                    plot_hist_2d(axes[i, j],data_sub,data_max,label_values[1],fig,False,latex_flag= True,cmap_limit=cmap_limit,xy_lim=[],vmax=0.5,font_size=8,label_size=8)
        plt.tight_layout()



def plot_distribution_comparison_2d(result_valid_set,sub_species_index = [1,2],data_index = [0,1,2]):
    """_summary_

    Args:
        result_valid_set (_type_): _description_
        sub_species_index (list, optional): _description_. Defaults to [1,2].
        data_index (list, optional): _description_. Defaults to [0,1,6].
    """    
    ssa_samples = np.array(result_valid_set['ssa_sample'])[:,:10000,:]
    nn_samples = np.array(result_valid_set['nn_sample'])[:,:10000,:]
    label_names = result_valid_set
    width_mm=183*1.1
    width_inch = width_mm * 0.0393701
    height_inch = width_inch * 0.6
    fig, axs = plt.subplots(2, 3, figsize=(width_inch, height_inch))
    plt.close(fig)
    fig = plt.figure(figsize=(width_inch,height_inch))
    gs = gridspec.GridSpec(2, 3)
    edgecolors = ["#df592a","#489bc5"]
    cmap_name = "Purples"
    cmap_limit = [0.1,0.8]
    fontsize,labelsize = 9,9

    random_perturbation = np.random.uniform(0, 1, ssa_samples.shape)
    ssa_samples=ssa_samples+random_perturbation
    nn_samples = nn_samples+random_perturbation
    xy_ticks = [[],[]]
    ssa_sample,nn_sample = ssa_samples[data_index[0],:,sub_species_index],nn_samples[data_index[0],:,sub_species_index]
    g0 = plot_jointplot(nn_sample[0,:],nn_sample[1,:],xy_ticks,axs[0, 0],lims=ssa_sample.max(axis=1),label_names = label_names,data_nn=True,color=edgecolors[0],cmap_name=cmap_name,cmap_limit=cmap_limit,fontsize = fontsize,labelsize=labelsize)
    g1 = plot_jointplot(ssa_sample[0,:],ssa_sample[1,:],xy_ticks,axs[0, 1],lims=ssa_sample.max(axis=1),label_names = label_names,color=edgecolors[0],cmap_name=cmap_name,cmap_limit=cmap_limit,fontsize = fontsize,labelsize=labelsize)

    ssa_sample,nn_sample = ssa_samples[data_index[1],:,sub_species_index],nn_samples[data_index[1],:,sub_species_index]
    g2 = plot_jointplot(nn_sample[0,:],nn_sample[1,:],xy_ticks,axs[0, 2],lims=ssa_sample.max(axis=1),label_names = label_names,data_nn=True,color=edgecolors[0],cmap_name=cmap_name,cmap_limit=cmap_limit,fontsize = fontsize,labelsize=labelsize)
    g3 = plot_jointplot(ssa_sample[0,:],ssa_sample[1,:],xy_ticks,axs[1, 0],lims=ssa_sample.max(axis=1),label_names = label_names,color=edgecolors[0],cmap_name=cmap_name,cmap_limit=cmap_limit,fontsize = fontsize,labelsize=labelsize)

    ssa_sample,nn_sample = ssa_samples[data_index[2],:,sub_species_index],nn_samples[data_index[2],:,sub_species_index]
    g4 = plot_jointplot(nn_sample[0,:],nn_sample[1,:],xy_ticks,axs[1, 1],lims=ssa_sample.max(axis=1),label_names = label_names,data_nn=True,color=edgecolors[0],cmap_name=cmap_name,cmap_limit=cmap_limit,fontsize = fontsize,labelsize=labelsize)
    g5 = plot_jointplot(ssa_sample[0,:],ssa_sample[1,:],xy_ticks,axs[1,2],lims=ssa_sample.max(axis=1),label_names = label_names,color=edgecolors[0],cmap_name=cmap_name,cmap_limit=cmap_limit,fontsize = fontsize,labelsize=labelsize)

    mg0 = SeabornFig2Grid(g1, fig, gs[0])
    mg1 = SeabornFig2Grid(g3, fig, gs[1])
    mg2 = SeabornFig2Grid(g5, fig, gs[2])
    mg3 = SeabornFig2Grid(g0, fig, gs[3])
    mg4 = SeabornFig2Grid(g2, fig, gs[4])
    mg5 = SeabornFig2Grid(g4, fig, gs[5])
    gs.tight_layout(fig)