import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class HellingerLoss(nn.Module):
    def __init__(self):
        super(HellingerLoss, self).__init__()

    def forward(self, p_pred, p_true):
        
        p_pred = F.softmax(p_pred, dim=-1)
        p_true = F.softmax(p_true, dim=-1)

        term1 = torch.sqrt(p_pred)
        term2 = torch.sqrt(p_true)
        loss = torch.sqrt(torch.sum((term1 - term2) ** 2) / 2)

        return loss



def scale_grad_val(grad_val,grad_thresh=0.005):
    try:
        min_grad_val,max_grad_val = 0.01,5
        # min_grad_val,max_grad_val = 0.001,5

        if  grad_val>=max_grad_val:
            return max_grad_val
        elif grad_val <min_grad_val and grad_val>0:
            return  min_grad_val
        elif grad_val > -min_grad_val and grad_val<0:
            return  -min_grad_val
        
        # elif grad_val < -max_grad_val and grad_val<0:
        #     return  -max_grad_val
        return grad_val*grad_thresh
    except Exception as e:
        print(f"error: {e}")
        return 1

def generate_nn_input_observed_data(r0,args,observed_data,net,observed_data_index=[2,3]):
    init = torch.as_tensor(args.initial_value, dtype=args.default_dtype_torch, device=args.device)
    rate = torch.tensor(r0, dtype=args.default_dtype_torch, device=args.device)

    prompt = torch.cat((torch.log(rate.view(-1)), init.view(-1)), dim=0)
    prompt = prompt.repeat(args.batch_size, 1)
    with torch.no_grad():
        _, sample = net.sample(prompt)
    observed_data =  torch.tensor(observed_data, dtype=args.default_dtype_torch, device=args.device)
    for i in range(len(observed_data_index)):
        sample[:,observed_data_index[i]] = observed_data[:,i]
    y_label = torch.as_tensor(sample, dtype=torch.long, device=args.device)
    y_label.to(args.device)
    osp = torch.cat((prompt, sample), dim=1)
    sp = torch.cat((prompt, sample), dim=1)
    temp_sp = sp.detach().cuda()
    return sample,y_label,osp,sp,temp_sp



def generate_nn_input_observed_data2(r0,args,observed_data,net,observed_data_index=[2,3],unobserved_number=0):
    init = torch.as_tensor(args.initial_value, dtype=args.default_dtype_torch, device=args.device)
    rate = torch.tensor(r0, dtype=args.default_dtype_torch, device=args.device)

    prompt = torch.cat((torch.log(rate.view(-1)), init.view(-1)), dim=0)
    prompt = prompt.repeat(args.batch_size, 1)
    observed_data =  torch.tensor(observed_data, dtype=args.default_dtype_torch, device=args.device)

    # unoberved_number = args.num_species - len(observed_data_index)
    if unobserved_number==0:
        sample = observed_data
    elif unobserved_number==1:
        labels = torch.cat([torch.zeros(observed_data.shape[0], 1, device=args.device), torch.ones(observed_data.shape[0], 1, device=args.device)], dim=0)
        sample = torch.cat([labels,observed_data.repeat(2, 1)], dim=1)
    elif unobserved_number==2:

        n=4*observed_data.shape[0]
        labels = torch.zeros(n, 2, device=args.device)

        labels[:n//4] = torch.tensor([0, 0], device=args.device)  
        labels[n//4:2*n//4] = torch.tensor([0, 1], device=args.device)  
        labels[2*n//4:3*n//4] = torch.tensor([1, 0], device=args.device) 
        labels[3*n//4:] = torch.tensor([1, 1], device=args.device)
        sample = torch.cat([labels,observed_data.repeat(4, 1)], dim=1)

    y_label = torch.as_tensor(sample, dtype=torch.long, device=args.device)
    y_label.to(args.device)
    osp = torch.cat((prompt, sample), dim=1)
    sp = torch.cat((prompt, sample), dim=1)
    temp_sp = sp.detach().cuda()
    return sample,y_label,osp,sp,temp_sp




def calculate_grad(r0,args,net,optimizer,param_num,grad_thresh,y_label1=[],sp1=[],sample_flag = False,observed_data=[],species_index=[],loss_weight=[],loss_type="Hellinger",unobserved_number = 1):
    optimizer.zero_grad()
    # old_sp1 = sp1.detach()
    sp1 = sp1.requires_grad_()
    # optimizer = torch.optim.Adam([sp1], lr=0.0001) 
    optimizer = torch.optim.Adam([sp1], lr=args.inference_lr) # new

    if len(species_index)>0:
        y_label1 = y_label1[:,species_index]
    # loss_cross1 = net(sp1,y_label1,species_index=species_index)
    if args.loss_type=="Likelihood":
        log_flag=True
    elif args.loss_type=="Hellinger":
        log_flag=False

    log_prob = net.log_joint_prob(sp1,log_flag=log_flag,unobserved_number=unobserved_number)
    if len(loss_weight)>0:
        pass
    # loss_weight =  torch.tensor(loss_weight, dtype=args.default_dtype_torch, device=args.device).repeat(2)
    loss_weight =  torch.tensor(loss_weight, dtype=args.default_dtype_torch, device=args.device)

    if args.loss_type=="Likelihood":
        loss_cross1 = (-log_prob*loss_weight).sum()
    elif args.loss_type == "Hellinger":
        hellinger_loss_fn = HellingerLoss()
        loss_cross1 = hellinger_loss_fn(log_prob,loss_weight)
    else:
        hellinger_loss_fn = HellingerLoss()
        loss_cross1 = hellinger_loss_fn(log_prob,loss_weight)

    if len(observed_data)>0:
        epsilon = 1e-15
        observed_data_prob = get_numpy_prob(observed_data.astype("int"))
        log_prob = net.log_joint_prob(sp1)
        log_Tp_t = np.array([observed_data_prob.get(tuple(row),epsilon) for row in y_label1[:,[1,2]]])
        log_Tp_t = torch.as_tensor(log_Tp_t, dtype=args.default_dtype_torch, device=args.device)
        prob = torch.exp(log_prob.detach())
        r_prob = torch.exp(log_Tp_t.detach())
        r_prob = r_prob / r_prob.sum()
        r_prob = (r_prob * prob.sum()).detach()
        loss = log_prob - log_Tp_t.detach()
        # loss_cross1 = torch.mean((loss - loss.mean()) * log_prob)
        # follow is HellingerLoss
        hellinger_loss_fn = HellingerLoss()
        loss_cross1 = hellinger_loss_fn(log_prob,log_Tp_t)

    loss_cross1.backward(retain_graph=False)
    optimizer.step() # new
    sp1.grad[:, param_num:] = 0 
    grad_means1 = sp1.grad.mean(dim=0)
    new_grad_means1 = []
    for index, value in enumerate(grad_means1):
        if index<len(r0):
            new_grad_means1.append(scale_grad_val(value,grad_thresh[index]))
        else:
            new_grad_means1.append(scale_grad_val(value))
    
    new_grad_means1 = torch.tensor(new_grad_means1, device=args.device)
    # optimizer.step()
    optimizer.zero_grad()

    sp1.grad=None

    # new_sp1 = sp1.detach()
    # new_sp1[:, param_num:] = old_sp1[:, param_num:]
    # new_grad_means1 = (old_sp1 - new_sp1).mean(dim=0)
    # new_grad_means1 = torch.tensor(new_grad_means1, device=args.device)
    del sp1
    del grad_means1
    del y_label1
    del optimizer
    torch.cuda.empty_cache()
    return new_grad_means1,loss_cross1


def load_synthetic_data(file_path,params_number=10,sample_number=3000,keep_index=[]):

    with open(file_path, 'r') as f:
        synthetic_datas = json.load(f)
    synthetic_samples_li = []
    for i in range(params_number):
        prompt_prob = synthetic_datas[i]
        sample_weight = np.array(list(prompt_prob[1].values()))
        sample_weight = sample_weight/sum(sample_weight)
        samples_str = np.random.choice(list(prompt_prob[1].keys()),sample_number,p=sample_weight)
        synthetic_samples = np.array([el.split("_") for el in samples_str ]).astype(int)
        synthetic_samples_li.append(synthetic_samples)
    if len(keep_index)>0:
        synthetic_samples_np = np.array(synthetic_samples_li)[:,:,keep_index]
    else:
        synthetic_samples_np = np.array(synthetic_samples_li)
    return synthetic_samples_np



# def nn_sample(promt_prob,net,para_indexs=range(0,100,10),bs=1000,true_param=True,sample_time = 50):
#     met_samples = []
#     for para_index in para_indexs:
#         print(" the sample index is",para_index)
#         met_times = []
#         with torch.no_grad():
#             if true_param:
#                 prompt = torch.as_tensor(promt_prob[para_index][0], dtype=args.default_dtype_torch, device=args.device)
#             else:
#                 prompt = torch.as_tensor(promt_prob[para_index], dtype=args.default_dtype_torch, device=args.device)

#             prompt = prompt.repeat(bs, 1)
#             samples = []
#             for i in range(sample_time):  # sampling sample_time x batch_size samples for each time point
#                 _, sample = net.sample(prompt)
#                 samples.append(sample.detach().cpu().numpy())
#                 del sample
#             met_samples.append(np.concatenate(samples, axis=0))
            
#     nn_samples=np.array(met_samples)
#     return nn_samples

# def augment_data(data):
#     if data.shape[0] % 1000 !=0:
#         additional_rows = 1000 - (data.shape[0]% 1000)
#         additional_data = data[np.random.choice(data.shape[0], additional_rows, replace=True)]
#         augmented_data = np.vstack([data, additional_data])
#     else:
#         augmented_data = data
#     return augmented_data

def augment_data(data,data_number = 3000):
    if data.shape[0] % data_number != 0:
        additional_rows = data_number - (data.shape[0] % data_number)
        # Randomly select and replicate a few rows
        additional_data = data[np.random.choice(data.shape[0], additional_rows, replace=True)]
        # Combine the original data with the additional rows
        augmented_data = np.vstack([data, additional_data])
    else:
        augmented_data = data
    return augmented_data

def get_numpy_prob(arr):
    # Count the frequency of each row
    unique_rows, counts = np.unique(arr, axis=0, return_counts=True)

    # Calculate the probabilities
    probabilities = counts / counts.sum()

    # Combine the unique rows and their corresponding probabilities into a dictionary
    result_dict = {tuple(row): prob for row, prob in zip(unique_rows, probabilities)}

    return result_dict

def judge_update_loss(numbers):
    if len(numbers) < 10:
        return True
    last_number = numbers[-1]
    previous_three = numbers[-10:-1]

    # Calculate the difference between the last number and the previous three numbers
    differences = [prev - last_number for prev in previous_three]

    # Check if any difference is less than 0.01
    has_decreased_less_than_threshold = any(diff < 0.01 for diff in differences)
    return has_decreased_less_than_threshold

class HellingerDistanceLoss(nn.Module):
    def __init__(self):
        super(HellingerDistanceLoss, self).__init__()

    def forward(self, p, q):
        return torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=-1)) / torch.sqrt(torch.tensor(2.0))

def get_model_config(model_name,args):
    if model_name == "afl":
        rub = torch.log(torch.tensor([2, 0.1, 10, 100], device=args.device))
        rlb = torch.log(torch.tensor([1e-8, 1e-8, 1e-8, 1e-8], device=args.device))
        rub_val,rlb_val = np.array([2, 0.1, 10, 100]),np.array([0.001, 0.001, 0.001, 0.001])
        r0 = [1.2, 0.05, 5, 50]
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["G", "M" ]
        observed_data_index = [1]
        observed_sample_index = [1]
    elif model_name == "arl":
        rub_li,rlb_li = [ 2, 5, 2,  1, 100,1],[ 0.01, 0.01, -2,  0.01, 0.01,0.1]
        rub_li,rlb_li = [ 2, 5, 4,  1, 40,1],[ 0.1,0, 0,  0.1, 0.1,0.1]
        rub = torch.log(torch.tensor(rub_li, device=args.device))
        rlb = torch.log(torch.tensor(rlb_li, device=args.device))
        rub_val,rlb_val = np.array(rub_li),np.array(rlb_li)

        r0 = np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["G", "Protein" ]
        observed_data_index = [1]
        observed_sample_index = [1]
    elif model_name == "isc":
        rub_val,rlb_val = np.array([20, 1, 1]),np.array([0.1, 0.1, 0.1])
        rub = torch.log(torch.tensor([20, 1, 1], device=args.device))
        rlb = torch.log(torch.tensor([0.1, 0.1, 0.1], device=args.device))
        r0 = np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["X"+"_"+str(i) for i in range(1,11)]
        observed_data_index = [0,1,2,3,4,5,6,7,8,9]
        observed_sample_index = [0,1,2,3,4,5,6,7,8,9]
    elif model_name == "toggle_switch":
        rub_val,rlb_val = np.array([5,10,5,5,5,30,30,0.4,0.4]),np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.1,0.1])
        rub = torch.log(torch.tensor(rub_val, device=args.device))
        rlb = torch.log(torch.tensor(rlb_val, device=args.device))
        r0= np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["G_A", "G_B","P_1","P_2" ]
        observed_data_index = [2,3]
        observed_sample_index = [2,3]

    elif model_name == "on_off_nm":
        rub_val,rlb_val = np.array([5,5,40,1,0.5]),np.array([0.01,0.01,0.01,0.1,0.1])
        rub = torch.log(torch.tensor(rub_val, device=args.device))
        rlb = torch.log(torch.tensor(rlb_val, device=args.device))
        r0= np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [0.1 for i in range(len(r0))]
        x_labels = ["G", "N","M" ]
        observed_data_index = [1,2]
        observed_sample_index = [1,2]
    
    elif model_name == "nm_nm":
        rub_val,rlb_val = np.array([5, 5, 5, 5, 5, 5, 5, 5, 30.0]),np.array([0.01, 0.01, 0.01, 0.01,0.01,0.01,0.01,0.01,1])
        rub = torch.log(torch.tensor([5, 5, 5, 5, 5, 5, 5, 5, 30.0], device=args.device))
        rlb = torch.log(torch.tensor([0.01, 0.01, 0.01, 0.01,0.01,0.01,0.01,0.01,1], device=args.device))
        r0= np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["G_u", "N","M" ]
        observed_data_index = [1,2]
        observed_sample_index = [1,2]

    elif model_name == "sir":
        rub_val,rlb_val = np.array([10, 10, 10, 0.05, 0.1, 0.2,  0.4, 0.2,0.2,0.2]),np.array([1, 1, 1, 0.02,0.02,0.1,0.2,0.1,0.1,0.1])
        rub = torch.log(torch.tensor([10, 10, 10, 0.05, 0.1, 0.2,  0.4, 0.2,0.2,0.2], device=args.device))
        rlb = torch.log(torch.tensor([1, 1, 1, 0.02,0.02,0.1,0.2,0.1,0.1,0.1], device=args.device))
        r0 = np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["S", "I","R" ]
    elif model_name == "sdp":
        rub_val,rlb_val = np.array([20, 0.2, 0.1, 0.1, 0.1, 0.3, 0.3]),np.array([1.0,0.05, 0.01,0.01,0.01,0.1,0.1])
        rub = torch.log(torch.tensor(rub_val, device=args.device))
        rlb = torch.log(torch.tensor(rlb_val, device=args.device))
        r0 = np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["S", "D" ]
    
    elif model_name == "ts_txl":

        rub_val,rlb_val = np.array([0.1, 0.1, 10, 10, 2, 0.5, 0.4]),np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2])
        rub = torch.log(torch.tensor(rub_val, device=args.device))
        rlb = torch.log(torch.tensor(rlb_val, device=args.device))
        r0= np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["G", "mRNA","Protein" ]
        observed_data_index = [1,2]
        observed_sample_index = [1,2]

    elif model_name == "sirs":
        rub_val,rlb_val = np.array([10, 0.2, 0.1,   0.4, 0.2,0.2,0.2]),np.array([0.05, 0.1,0.05,0.05,0.05,0.05,0.05])
        rub = torch.log(torch.tensor(rub_val, device=args.device))
        rlb = torch.log(torch.tensor(rlb_val, device=args.device))
        r0 = np.random.uniform(low=rlb_val, high=rub_val)
        grad_thresh = [1 for i in range(len(r0))]
        x_labels = ["S", "I","R" ]
        observed_sample_index = [0,1,2]
        observed_data_index = [0,1,2]
    return rub_val,rlb_val,rub,rlb,r0,grad_thresh,x_labels,observed_data_index,observed_sample_index


def nn_sample(promt_prob,net,args,param_indexs=range(0,100,10),bs=1000,true_param=True,sample_time = 50):
    samples_all = []
    for param_index in param_indexs:
        met_times = []
        with torch.no_grad():
            if true_param:
                prompt = torch.as_tensor(promt_prob[param_index][0], dtype=args.default_dtype_torch, device=args.device)
            else:
                prompt = torch.as_tensor(promt_prob[param_index], dtype=args.default_dtype_torch, device=args.device)

            prompt = prompt.repeat(bs, 1)
            samples = []
            for i in range(sample_time):  # sampling sample_time x batch_size samples for each time point
                _, sample = net.sample(prompt)
                samples.append(sample.detach().cpu().numpy())
                del sample
            samples_all.append(np.concatenate(samples, axis=0))  
    nn_samples=np.array(samples_all)
    return nn_samples


def get_max_density_point(data,weights):
    # Initialize the kernel density estimator
    weights =weights/sum(weights)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
    # Fitting the data using weights
    kde.fit(data, sample_weight=weights)
    log_dens = kde.score_samples(data)
    densities = np.exp(log_dens)
    max_density_index = np.argmax(densities)
    max_density_point = data[max_density_index]
    return max_density_point

def judge_density(array,lst):
    row_max = array.max(axis=0)
    row_min = array.min(axis=0)
    lst = lst[:len(row_min)]
    result = [(row_min[i] <= lst[i] <= row_max[i]) for i in range(len(lst))]
    return all(result)