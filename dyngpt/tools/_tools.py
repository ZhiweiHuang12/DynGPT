import itertools
import numpy as np
from scipy.stats import entropy
from scipy.special import rel_entr
import json
import os
import torch
import pkg_resources
import pandas as pd
from dyngpt.dynmodels.default_model import get_default_model_config
from dyngpt._nnmodel._nnmodel import NNmodel_FT
from dyngpt._utils._util import *

def update_default_config(config_dict):
    """Update the basic configuration with values from the default configuration.

    Args:
        config_dict (dict): A dictionary containing configuration parameters.

    Returns:
        Namespace: An updated configuration object with modified parameters.
    """  
    args = get_default_model_config()
    args.model = config_dict.get("model",args.model)
    args.num_species = config_dict.get("num_species",args.num_species)  # Species number
    args.num_reactions = config_dict.get("num_reactions",args.num_reactions)  # reaction number
    args.num_init =  config_dict.get("num_init",args.num_init)
    # Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation.
    args.state_upper_bound = config_dict.get("state_upper_bound",args.state_upper_bound)
    args.constrains = config_dict.get("constrains",args.constrains)
    args.constrains = np.array(args.constrains)
    # --------------------------------- neural network model architecture  -------------------------------------#
    args.variable_dimension = config_dict.get("variable_dimension",args.variable_dimension)  # dimensionality of the variable matrix including prompt and states
    args.embedding_dimension = config_dict.get("embedding_dimension",args.embedding_dimension)  # transformer emb_dim
    args.feed_forward_dimension = config_dict.get("feed_forward_dimension",args.feed_forward_dimension) # transformer ff_dim
    args.num_encoder_layers = config_dict.get("num_encoder_layers",args.num_encoder_layers)  # transformer n_layer
    args.n_head = config_dict.get("n_head",args.n_head)  # transformer n_head
    args.block_size = config_dict.get("block_size",args.block_size)  # maximum input length for dyngpt
    args.lr =  config_dict.get("lr",args.lr)  # initial learning rate
    args.batch_size = config_dict.get("batch_size",args.batch_size) 
    args.bias = config_dict.get("bias",args.bias)   # False for training dyngpt
    args.dropout_rate =  config_dict.get("dropout_rate",args.dropout_rate)  # for dyngpt
    args.weight_decay = config_dict.get("weight_decay",args.weight_decay)  # for configure dyngpt optimizer
    args.beta1 = config_dict.get("beta1",args.beta1)  # for configure dyngpt optimizer
    args.beta2 = config_dict.get("beta2",args.beta2)  # for configure dyngpt optimizer
    args.decay_lr = config_dict.get("decay_lr",args.decay_lr)  # whether to decay the learning rate
    args.epochs = config_dict.get("epochs",args.epochs)  # usually should be 5000-10000 epochs for convergent training
    args.start_epoch = config_dict.get("start_epoch",args.start_epoch) # changed when loading pretrain dyngpt state file
    args.last_epoch = config_dict.get("last_epoch",args.last_epoch)  # specify last epoch for loading dyngpt pretrain state file
    args.initial_value = config_dict.get("initial_value",args.last_epoch)
    
    # inference
    args.rlb_val = config_dict.get("rlb_val",[])
    args.rub_val = config_dict.get("rub_val",[])
    args.state_name = config_dict.get("state_name",[])
    args.observed_data_index = config_dict.get("observed_data_index",[])
    # file_path
    args.load_weight = False
    args.weighted_sample = True
    args.device = torch.device("cuda")
    args.result_dir = config_dict.get("result_dir","") 
    args.figure_dir = config_dict.get("figure_dir","") 
    os.makedirs(args.result_dir ,exist_ok=True)
    os.makedirs(args.figure_dir,exist_ok=True)
    return args


def update_dynmodel_config(config_basic,config_path_dynmodel):
    """Update the basic configuration with parameters from a dynamic model configuration file.

    Args:
        config_basic (dict): The basic configuration dictionary.
        config_path_dynmodel (str): Path to the JSON file containing dynamic model configuration.

    Returns:
        dict: The updated configuration dictionary.
    """  
    with open(config_path_dynmodel, 'r') as file:
        dynmodel_config = json.load(file)
    config_basic["model"] = dynmodel_config["model_name"]
    config_basic["num_species"] = len(dynmodel_config["minimal_state_indices"])
    config_basic["num_reactions"] = len(dynmodel_config["parameters"]["param_names"])
    config_basic["num_init"] = len(dynmodel_config["minimal_state_indices"])
    minimal_state_indices = [i-1 for i in dynmodel_config["minimal_state_indices"]]
    config_basic["initial_value"] =  np.array(dynmodel_config["initial_val"])[minimal_state_indices]
    config_basic["initial_value"] = np.append(config_basic["initial_value"], 0)
    config_basic["state_upper_bound"] =  int(max(dynmodel_config["state_upper_bound"]))
    config_basic["constrains"] = dynmodel_config["state_upper_bound"]
    config_basic["rlb_val"] = dynmodel_config["parameters"]["lower_bound_val"]
    config_basic["rub_val"] = dynmodel_config["parameters"]["upper_bound_val"]
    config_basic["state_name"] = np.array(dynmodel_config["state_names"])[minimal_state_indices]
    config_basic["observed_data_index"] = [i-1 for i in dynmodel_config["observed_data_indices"]]
    return config_basic


def load_synthetic_data(file_path,params_number=10,sample_number=3000,keep_index=[]):
    """Load and process synthetic data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing synthetic data.
        params_number (int, optional): Number of parameter sets. Defaults to 10.
        sample_number (int, optional): Number of samples per parameter set. Defaults to 3000.
        keep_index (list, optional): Indices of features to keep in the output. Defaults to an empty list.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Processed synthetic samples.
            - np.ndarray: Corresponding parameter values.
    """   
    with open(file_path, 'r') as f:
        synthetic_datas = json.load(f)
    synthetic_samples_li = []
    params_li = []
    for i in range(params_number):
        prompt_prob = synthetic_datas[i]
        sample_weight = np.array(list(prompt_prob[1].values()))
        sample_weight = sample_weight/sum(sample_weight)
        samples_str = np.random.choice(list(prompt_prob[1].keys()),sample_number,p=sample_weight)
        synthetic_samples = np.array([el.split("_") for el in samples_str ]).astype(int)
        synthetic_samples_li.append(synthetic_samples)
        params_li.append(prompt_prob[0])
    if len(keep_index)>0:
        synthetic_samples_np = np.array(synthetic_samples_li)[:,:,keep_index]
    else:
        synthetic_samples_np = np.array(synthetic_samples_li)
    return synthetic_samples_np,np.array(params_li)

def compute_sampling_stats(args,param_index = range(100)):
    """Compute statistical metrics for sampled data.

    Args:
        args (Namespace): Configuration object containing model parameters.
        param_index (range, optional): Indices of parameters to process. Defaults to range(100).

    Returns:
        dict: A dictionary containing computed statistics, including mean, standard deviation, 
              KL divergence, and sampled data.
    """  
    # x_labels = 
    weight_path= args.weight_nn_fine_tune_path
    x_labels = args.state_name
    device = args.device
    net = NNmodel_FT(**vars(args))
    net.to(device)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    CUDA_LAUNCH_BLOCKING=1
    promt_prob = load_promt_prob(args,file_path = args.valid_dataset_path)
    params = np.array([promt_prob[i][0] for i in param_index])
    state = torch.load(weight_path,map_location=torch.device(args.device))
    net.load_state_dict(state["net_state_dict"])
    net.to(args.device)
    net.eval()
    samples = nn_sample(args,promt_prob,net,param_index)
    
    mean_val_nn = get_stat(samples,sta_type = "mean")
    std_val_nn = get_stat(samples,sta_type = "std")
    prob_val_nn = get_prob(samples)
    
    mean_val_ssa,std_val_ssa,prob_val_ssa = get_stat_prob_ssa(args,param_index)
    kl_div_result = []
    for i in range(len(prob_val_nn)):
        kl_div_result.append([kl_div(prob_val_nn[i][j],prob_val_ssa[i][j]) for j in range(len(prob_val_nn[0]))])
    kl_div_np = np.vstack(kl_div_result)
    if not os.path.isdir(args.out_dir+"data/"):
        os.mkdir(args.out_dir+"data/")
    pd.DataFrame(kl_div_np,columns=x_labels).to_csv(args.out_dir+"data/kl_{}.csv".format(args.model),index=False)

    save_stats_data(mean_val_nn,std_val_nn,x_labels,args,"nn")
    save_stats_data(mean_val_ssa,std_val_ssa,x_labels,args,"ssa")
    indices = np.where(np.all(kl_div_np < 100, axis=1))[0] 
    print("the indices length is",len(indices))
    
    all_samples_li = sample_joint_prob_all(args,param_index,indices)
    save_counts_data_all(all_samples_li,samples[indices],x_labels,args)

    params = params[indices]
    np.save(args.out_dir+"data/params_{}.npy".format(args.model), params)

    result_dict = {}
    result_dict["ssa_sample"] = all_samples_li
    result_dict["nn_sample"] =  samples[indices]
    result_dict["mean_val_ssa"] =  pd.DataFrame(mean_val_ssa,columns=args.state_name)
    result_dict["std_val_ssa"] =  pd.DataFrame(std_val_ssa,columns=args.state_name)
    result_dict["mean_val_nn"] = pd.DataFrame(mean_val_nn,columns=args.state_name)
    result_dict["std_val_nn"] = pd.DataFrame(std_val_nn,columns=args.state_name)
    result_dict["kl_div"] = pd.DataFrame(kl_div_np,columns=args.state_name)
    result_dict["model_name"] = args.model

    return result_dict

def compute_kl_stats(result_infer):
    """Calculate statistical metrics and the Kullback-Leibler divergence between neural network-generated samples
        and observed data.

    Args:
        result_infer (dict): A dictionary containing inference results with the following keys:
            - "nn_sample_datas" (list or ndarray): Samples generated by the neural network.
            - "observed_datas" (list or ndarray): Observed real-world sample data.
            - "state_name" (list of str): List of state variable names.
            - "model" (str): Name of the model used for inference.

    Returns:
        dict: A dictionary containing computed statistical metrics and KL divergence:
            - "ssa_sample" (ndarray): Observed sample data.
            - "nn_sample" (ndarray): Neural network-generated sample data.
            - "mean_val_ssa" (pd.DataFrame): Mean values of observed data with state names as columns.
            - "std_val_ssa" (pd.DataFrame): Standard deviations of observed data.
            - "mean_val_nn" (pd.DataFrame): Mean values of neural network samples.
            - "std_val_nn" (pd.DataFrame): Standard deviations of neural network samples.
            - "kl_div" (pd.DataFrame): KL divergence values for each state variable.
            - "model_name" (str): Model name used for inference.
    """      
    nn_samples = result_infer["nn_sample_datas"]
    mean_val_nn = get_stat(nn_samples,sta_type = "mean")
    std_val_nn = get_stat(nn_samples,sta_type = "std")
    prob_val_nn = get_prob(nn_samples)

    ssa_samples = result_infer["observed_datas"]
    mean_val_ssa = get_stat(ssa_samples,sta_type = "mean")
    std_val_ssa = get_stat(ssa_samples,sta_type = "std")
    prob_val_ssa = get_prob(ssa_samples)

    kl_div_result = []
    for i in range(len(prob_val_nn)):
        kl_div_result.append([kl_div(prob_val_nn[i][j],prob_val_ssa[i][j]) for j in range(len(prob_val_nn[0]))])
    kl_div_np = np.vstack(kl_div_result)

    result_dict = {}
    result_dict["ssa_sample"] = ssa_samples
    result_dict["nn_sample"] =  nn_samples
    result_dict["mean_val_ssa"] =  pd.DataFrame(mean_val_ssa,columns=result_infer["state_name"])
    result_dict["std_val_ssa"] =  pd.DataFrame(std_val_ssa,columns=result_infer["state_name"])
    result_dict["mean_val_nn"] = pd.DataFrame(mean_val_nn,columns=result_infer["state_name"])
    result_dict["std_val_nn"] = pd.DataFrame(std_val_nn,columns=result_infer["state_name"])
    result_dict["kl_div"] = pd.DataFrame(kl_div_np,columns=result_infer["state_name"])
    result_dict["model_name"] = result_infer["model"]
    return result_dict