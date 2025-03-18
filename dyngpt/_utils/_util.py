import itertools
import numpy as np
from scipy.stats import entropy
from scipy.special import rel_entr
import json
import os
import pkg_resources
import pandas as pd
import torch


def calculate_kl_divergence(data1,data2):
    data1 = data1.astype("int")
    data2 = data2.astype("int")
    max_val = data1.max(axis=0)
    unique_rows1, counts1 = np.unique(data1, axis=0, return_counts=True)
    freq1 = {tuple(row): count for row, count in zip(unique_rows1, counts1)}

    unique_rows2, counts2 = np.unique(data2, axis=0, return_counts=True)
    freq2 = {tuple(row): count for row, count in zip(unique_rows2, counts2)}

    all_keys1 = set(freq1.keys()).union(set(freq2.keys()))
    if len(max_val) == 2:
        all_keys2 = list(itertools.product(range(int(max_val[0])), range(int(max_val[1]))))
    elif len(max_val)>2:
        all_keys2 = all_keys1
    else:
        all_keys2 = [(i,) for i in range(int(max_val[0]))]
    
    all_keys = set(all_keys1) & set(all_keys2)

    p = np.array([freq1.get(k, 0.0) for k in all_keys], dtype=np.float64)
    q = np.array([freq2.get(k, 0.0) for k in all_keys], dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    epsilon = 1e-15
    q = q + epsilon
    p = p + epsilon
    kl_divergence = np.sum(rel_entr(p, q))
    return kl_divergence

def load_promt_prob(args,data_type="valid",file_path=""):
    if len(file_path)==0:
        file_path = pkg_resources.resource_filename(__name__, 'data/{}_{}_stable.json'.format(args.model,data_type))
    # file_path = args.data_dir+"{}_{}_stable.json".format(args.model,data_type)
    with open(file_path, 'r') as f:
        s1 = json.load(f)
    return s1

def ensure_dir(filename):
    """
    make sure directory exists
    """
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass

def print_memory():
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    # print(f"Allocated GPU memory: {allocated_memory:.4f} GB")


######################
def calculate_probabilities(data):
    """
    Calculate the probability distribution of each integer in the list
    """
    total_count = len(data)
    value_counts = pd.Series(data).value_counts(normalize=True).sort_index()
    return value_counts

def get_Aligned_prob(prob1,prob2):
    df = pd.DataFrame({
        'Value': prob1.index.union(prob2.index), 
        'List1_Probability': prob1.reindex(prob1.index.union(prob2.index), fill_value=0),
        'List2_Probability': prob2.reindex(prob1.index.union(prob2.index), fill_value=0)
    })
    df = df[df["Value"]<=max(prob1.index)]
    df['List1_Probability']=df['List1_Probability']/(df['List1_Probability'].sum())
    df['List2_Probability']=df['List2_Probability']/(df['List2_Probability'].sum())
    return df.iloc[:,[1,2]]

def get_ssa_prob(prompt_prob_datasets,data_index):
    origin_prob =[np.hstack([key.split("_"),[value]]) for key, value in prompt_prob_datasets[data_index][1].items()]
    prob_df = pd.DataFrame(origin_prob,columns=["s0","s1","counts"]).astype(int)
    prob_df["prob"] = prob_df.iloc[:,-1]/prob_df.iloc[:,-1].sum()
    ssa_prob = prob_df.groupby("s1")["prob"].sum()
    return ssa_prob

# following function from utils_nn
# def load_promt_prob(args):
#     file_path = args.data_dir+"{}_{}_stable.json".format(args.model,args.data_type)
#     with open(file_path, 'r') as f:
#         s1 = json.load(f)
#     return s1

def get_stat(samples,sta_type = "mean"):
    val_list = []
    for i in range(samples.shape[-1]):
        temp_sample = samples[:, :, i]
        if sta_type == "mean":
            val_list.append(np.mean(temp_sample, 1))
        elif sta_type == "std":
            val_list.append(np.std(temp_sample, 1))
    result = np.vstack(val_list).transpose()
    return result

def get_prob(samples):
    prob_li = []
    for i in range(samples.shape[0]):
    # for i in range(test_num):
        prob_i = [marginal_pro(samples[i, :, j]) for j in range(samples.shape[2])]
        prob_li.append(prob_i)
    return prob_li

def get_stat_prob_ssa(args,test_index,args_flag=True):
    # Load the data obtained from the ssa sample and calculate the mean variance and marginal probability
    if args_flag:
        # promt_prob = load_promt_prob(args)
        promt_prob = load_promt_prob(args,file_path = args.valid_dataset_path)
    else:
        promt_prob = args
    mean_val = []
    var_val = []
    prob_val = []
    # for para_index in range(samples.shape[0]):
    for para_index in test_index:
        origin_prob =[np.hstack([key.split("_"),[value]]) for key, value in promt_prob[para_index][1].items()]
        prob_df = pd.DataFrame(origin_prob).astype(int)
        prob_df["prob"] = prob_df.iloc[:,-1]/prob_df.iloc[:,-1].sum()
        
        mean_val.append([sum(prob_df["prob"]*prob_df.iloc[:,i]) for i in range(prob_df.shape[1]-2)])
        var_val.append([variance(prob_df.iloc[:,i],prob_df["prob"]) for i in range(prob_df.shape[1]-2)])
        prob_val.append([prob_df.groupby(col)['prob'].sum().to_dict() for col in prob_df.columns.tolist()[:-1]])
    
    mean_val = np.array(mean_val)
    var_val = np.array(var_val)
    std_val = np.sqrt(var_val)
    prob_val = prob_val
    return mean_val,std_val,prob_val

def marginal_pro(data):
    elements, counts = np.unique(data.astype(int), return_counts=True)
    probabilities = counts / len(data)
    probability_density = dict(zip(elements, probabilities))
    return probability_density

def variance(X, P):
    mu = sum([x*p for x, p in zip(X, P)])
    var = sum([p*(x-mu)**2 for x, p in zip(X, P)])
    return var


def kl_div(p,q):
    """
    Calculate the KL divergence of two distributions
    """
    keys = set(list(p.keys()) + list(q.keys()))
    epsilon = 1e-10
    p_aligned = np.array([p.get(key, epsilon) for key in keys])
    q_aligned = np.array([q.get(key, epsilon) for key in keys])
    kl_div = entropy(p_aligned, q_aligned)
    return kl_div


def sample_joint_prob(args,species_i,species_j,test_index,indices,size=100000):
    """
    i,j is the order of species
    """
    promt_prob = load_promt_prob(args)
    px_y_samples_np_li = []
    # for para_index in indices:
    for para_index in indices:
        temp_index = test_index[para_index]
        origin_prob =[np.hstack([key.split("_"),[value]]) for key, value in promt_prob[temp_index][1].items()]
        prob_df = pd.DataFrame(origin_prob).astype(int)
        prob_df["prob"] = prob_df.iloc[:,-1]/prob_df.iloc[:,-1].sum()
        prob_df["join_count"] = prob_df.iloc[:,species_i].astype(str)+"_"+prob_df.iloc[:,species_j].astype(str)
        prob_df_px_y = prob_df.groupby("join_count")['prob'].sum()
        px_y_samples = np.random.choice(prob_df_px_y.index, size=size, p=prob_df_px_y.values)
        px_y_samples_np = np.vstack([el.split("_") for el in px_y_samples]).astype(int)
        px_y_samples_np_li.append(px_y_samples_np)
    return px_y_samples_np_li

def sample_joint_prob_all(args,test_index,indices,size=100000):
    promt_prob = load_promt_prob(args,file_path = args.valid_dataset_path)
    samples_np_li = []
    for para_index in indices:
        # print("the count sample index is",para_index)
        temp_index = test_index[para_index]
        temp_data = pd.DataFrame(list(promt_prob[temp_index][1].items()), columns=['key', 'value'])
        temp_data["prob"] = temp_data.iloc[:,-1]/temp_data.iloc[:,-1].sum()
        samples = np.random.choice(temp_data["key"].tolist(), size=size, p=temp_data["prob"])
        samples_np = np.vstack([el.split("_") for el in samples]).astype(int)
        samples_np_li.append(samples_np)
    return samples_np_li

def nn_sample(args,promt_prob,net,para_indexs=range(0,100,10),bs=1000,file_path=""):
    # file_path = saved_data_path + "samples"
    met_samples = []
    for para_index in para_indexs:
        # print(" the index is",para_index)
        met_times = []
        with torch.no_grad():
            prompt = torch.as_tensor(promt_prob[para_index][0], dtype=args.default_dtype_torch, device=args.device)
            prompt = prompt.repeat(bs, 1)
            samples = []
            for i in range(100):  # sampling 100 x batch_size samples for each parameters
                _, sample = net.sample(prompt)
                samples.append(sample.detach().cpu().numpy())
            met_samples.append(np.concatenate(samples, axis=0))
    nn_samples=np.array(met_samples)
    # np.savez(file_path, samples=nn_samples)
    return nn_samples
    
def save_stats_data(mean_val_nn,std_val_nn,x_labels,args,data_type):
    mean_df = pd.DataFrame(mean_val_nn,columns=x_labels)
    mean_df.to_csv(args.out_dir+"data/mean_{}_{}.csv".format(data_type,args.model),index=False)
    std_df = pd.DataFrame(std_val_nn,columns=x_labels)
    std_df.to_csv(args.out_dir+"data/std_{}_{}.csv".format(data_type,args.model),index=False)

def save_counts_data_all(data1,data2,data_col,args):
    data_dir = args.out_dir + "data/joint_prob"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for i in range(len(data1)):
        temp_pd = pd.DataFrame(data1[i],columns=data_col)
        temp_pd.to_csv(args.out_dir + "data/joint_prob/joint_species_ssa_counts_{}_all.csv".format(i),index=False)
    for j in range(data2.shape[0]):
        temp_pd = pd.DataFrame(data2[j],columns=data_col).astype("int")
        temp_pd.to_csv(args.out_dir + "data/joint_prob/joint_species_nn_counts_{}_all.csv".format(j),index=False)

