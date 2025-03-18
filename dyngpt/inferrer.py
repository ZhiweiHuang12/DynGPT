import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc
import random
import time
import torch.optim as optim
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from dyngpt._nnmodel._nnmodel import NNmodel,NNmodel_FT
from dyngpt._utils._util import *
from dyngpt._utils._util_infer import *
from dyngpt._utils._util_plotting import *
from dyngpt.plotting._plotting import plot_afl_inference_data_burst_distribution
from dyngpt.plotting._plotting import plot_nm_nm_inference_data_burst_distribution
from dyngpt.plotting._plotting import plot_isc_compare_distribution

def inferring_dynamics(observed_data,config_infer,gene_names=[],true_params=[],synthetic_flag=0,new_model=False,synthetic_data=[]):
    """Infer the underlying dynamic parameters of a state transition network from observed data.

    Args:
        observed_data (numpy.ndarray): The observed dataset containing state counts.
        config_infer (Namespace): Configuration object containing inference parameters.
        gene_names (list, optional): List of gene names corresponding to the observed data. Defaults to [].
        true_params (list, optional): List of true parameter values for validation. Defaults to [].
        synthetic_flag (int, optional): Flag indicating whether synthetic data is used (1) or not (0). Defaults to 0.
        new_model (bool, optional): Whether to initialize a new model or use a pre-trained model. Defaults to False.
        synthetic_data (list, optional): Synthetic dataset for model validation, if applicable. Defaults to [].

    Returns:
        dict: Contains the inferred parameters, loss values, KL divergence values, and other evaluation metrics.
    """  
    truncated_index=-50
    filtered_observed_data = observed_data
    figure_dir, result_dir = config_infer.figure_dir,config_infer.result_dir
    args = config_infer
    test_params = 6
    test_params = min(test_params,filtered_observed_data.shape[0])
    iter_epoch = args.iteration_steps
    random_r0_number = args.random_r0_number
    model_name = args.model
    args.lr = args.inference_lr
    args.loss_type = args.inference_loss_func
    if model_name=="arl":
        lr_val = 1
    elif model_name == "on_off_nm":
        lr_val = 8
    else:
        lr_val = 1
    bs = 1000
    args.lr = lr_val
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    CUDA_LAUNCH_BLOCKING=1
    if new_model:
        rub_val,rlb_val = np.array(args.rub_val),np.array(args.rlb_val)
        rub = torch.log(torch.tensor(rub_val, device=args.device))
        rlb = torch.log(torch.tensor(rlb_val, device=args.device))
        r0= np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = args.state_name
        observed_data_index = args.observed_data_index
        observed_sample_index = args.observed_data_index
        weight_path = args.weight_nn_fine_tune_path
    else:
        rub_val,rlb_val,rub,rlb,r0,grad_thresh,x_labels,observed_data_index,observed_sample_index = get_model_config(model_name,args)
        weight_path = pkg_resources.resource_filename(__name__, 'weights/{}_final.pt'.format(args.model))

    param_num = args.num_reactions
    state = torch.load(weight_path)
    net = NNmodel_FT(**vars(args))
    net.load_state_dict(state["net_state_dict"])
    net.to(args.device)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    
    es_param_alls,es_loss_all_nps,es_param_all_nps,sub_filtered_observed_data_li = [],[],[],[]
    valid_gene_names=[]
    valid_true_params = []
    best_kl_val = []
    kl_true_data_li = []
    if len(gene_names)==0:
        gene_names = list(range(test_params))
    for k in range(test_params):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            data_number = args.batch_size//2
            if synthetic_flag:
                unobserved_number = 0
            else:
                unobserved_number = args.num_species - len(observed_data_index)
            if  unobserved_number==0:
                single_observed_data, counts = np.unique(filtered_observed_data[k], axis=0, return_counts=True)
                loss_weight = counts/sum(counts)
                sub_filtered_observed_data = single_observed_data    
                args.batch_size = sub_filtered_observed_data.shape[0] 
            else:
                if len(observed_sample_index)==1:     
                    single_observed_data, counts = np.unique(filtered_observed_data[k], axis=0, return_counts=True)
                    loss_weight = counts/sum(counts)
                    sub_filtered_observed_data = single_observed_data.T.reshape(len(single_observed_data),1)
                    args.batch_size  = len(single_observed_data)*2
                elif len(observed_sample_index)==2:
                    single_observed_data, counts = np.unique(filtered_observed_data[k], axis=0, return_counts=True)
                    loss_weight = counts/sum(counts)
                    sub_filtered_observed_data = single_observed_data
                    args.batch_size  = len(counts) * (args.num_species - len(observed_data_index))*2
                else:
                    single_observed_data, counts = np.unique(filtered_observed_data[k], axis=0, return_counts=True)
                    sub_filtered_observed_data = single_observed_data
                    args.batch_size  = len(counts) * (args.num_species - len(observed_data_index))*2
                    loss_weight = counts/sum(counts)

            es_param_all,es_loss_all,kl_divergence_val_li,best_param_li = [], [],[],[]
            optimizer = net.configure_optimizers(args.weight_decay, args.lr,(args.beta1, args.beta2),device_type=args.device)
            for l in range(random_r0_number):
                r0 = np.random.uniform(low=rlb_val, high=rub_val)
                if args.model=="arl":
                    r0[2]=r0[2]+2
                temp_loss = []
                temp_param = []
                x,y_label,osp,sp,temp_sp = generate_nn_input_observed_data2(r0,args,sub_filtered_observed_data,net,observed_data_index,unobserved_number=unobserved_number)
                update_flag = 0
                for i in range(iter_epoch):
                    new_grad_means, loss_val = calculate_grad(r0,args,net,optimizer,param_num=param_num,grad_thresh=grad_thresh, y_label1=y_label,sp1=sp,loss_weight=loss_weight,unobserved_number=unobserved_number)
                    loss_cross = loss_val
                    nsp = (sp.detach() - new_grad_means.detach()).detach()
                    for i_param in range(param_num):
                        lower_bound, upper_bound = rlb[i_param],rub[i_param]
                        nsp[:param_num,i_param] = torch.clamp(nsp[:param_num,i_param], min=lower_bound, max=upper_bound) 
                    
                    sp=nsp.detach().cuda()
                    temp_sp = nsp.detach()
                    
                    temp_loss_val = loss_cross.detach().cpu().numpy()
                    temp_param_val = temp_sp[0,:param_num].cpu().numpy()
                   
                    temp_param.append(temp_param_val)
                    temp_loss.append(temp_loss_val)
                    
                    if len(temp_loss)>20 and (temp_loss_val > min(temp_loss)):
                        update_flag = update_flag+1
                    if update_flag>1:
                        break

                    del loss_cross
                    del new_grad_means
                es_loss_all.extend(temp_loss[truncated_index:]) # -20
                es_param_all.extend(temp_param[truncated_index:])

                es_param = es_param_all[-1].reshape(1,es_param_all[-1].shape[0])
                es_param_prompt = np.hstack((es_param, np.tile(args.initial_value, (es_param.shape[0], 1))))
                
                if synthetic_flag:
                    if len(synthetic_data)>0:
                        kl_true_data = synthetic_data[k]
                    else:
                        kl_true_data = filtered_observed_data[k]
                else:
                    if len(observed_sample_index)==1:
                        kl_true_data = filtered_observed_data[k].reshape(filtered_observed_data[k].shape[0],1)
                    else:
                        kl_true_data = filtered_observed_data[k]
            del optimizer
            es_loss_all_np = np.hstack(es_loss_all)
            es_param_all_np = np.vstack(es_param_all)
            weights = np.exp(-es_loss_all_np)/np.sum(np.exp(-es_loss_all_np))
            best_param = get_max_density_point(es_param_all_np,weights) 
            best_param = best_param.reshape(1,best_param.shape[0])
            best_param_prompt = np.hstack((best_param, np.tile(args.initial_value, (best_param.shape[0], 1))))    
            best_param_samples = nn_sample(best_param_prompt,net,args=args, param_indexs=range(0,best_param.shape[0],1),bs=1000,true_param=False)
            if synthetic_flag:   
                kl_divergence_val = calculate_kl_divergence(kl_true_data,best_param_samples[0,:,:])
            else:
                kl_divergence_val = calculate_kl_divergence(kl_true_data,best_param_samples[0,:,observed_sample_index].T)
            best_kl_val.append(kl_divergence_val)
            es_param_alls.append(best_param.flatten())
            es_loss_all_nps.append(es_loss_all_np)
            es_param_all_nps.append(es_param_all_np)

            sub_filtered_observed_data_li.append(filtered_observed_data[k])
            kl_true_data_li.append(kl_true_data)
            valid_gene_names.append(gene_names[k])
            if len(true_params)>0:
                valid_true_params.append(true_params[k])
        except Exception as e:
            print(f"error:{e}")
            continue
        torch.cuda.empty_cache()

    observed_datas = np.array(sub_filtered_observed_data_li)
    kl_true_data_np = np.array(kl_true_data_li)
    es_params=np.array(es_param_all_nps)
    es_param = np.array(es_param_alls)
    es_param_prompt = np.hstack((es_param, np.tile(args.initial_value, (es_param.shape[0], 1))))
    nn_sample_datas = nn_sample(es_param_prompt,net,args=args,param_indexs=range(0,es_param.shape[0],1),bs=min(args.batch_size,1000),true_param=False)
    es_loss=np.array(es_loss_all_nps)
    valid_gene_names = np.array(valid_gene_names)
    valid_true_params = np.array(valid_true_params)
    best_kl_val = np.array(best_kl_val)

    if len(result_dir)>0:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        np.savez(result_dir +'inferring_result_{}.npz'.format(model_name), observed_datas=observed_datas,nn_sample_datas=nn_sample_datas,es_params=es_params,es_loss=es_loss,es_param=es_param,gene_names=valid_gene_names,kl_val=best_kl_val,true_params = valid_true_params,synthetic_data=kl_true_data_np)
        
    result_dict = {}
    result_dict["observed_datas"] = observed_datas
    result_dict["nn_sample_datas"] = nn_sample_datas
    result_dict["es_params"] = es_params
    result_dict["es_param"] = es_param
    result_dict["es_loss"] = es_loss
    result_dict["model"] = config_infer.model
    result_dict["state_name"] = config_infer.state_name
    return result_dict




