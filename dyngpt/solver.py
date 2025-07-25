import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
from dyngpt._nnmodel._nnmodel import NNmodel
from dyngpt._utils._util_infer import *
from dyngpt._utils._util_train import *
from dyngpt._utils._util import *
import math
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dyngpt._nnmodel._nnmodel import NNmodel
from dyngpt._nnmodel._nnmodel import NNmodel_FT

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def solving_state_transition_network(model_name,params=[],initial_value=[],args=""):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    CUDA_LAUNCH_BLOCKING=1
    weight_path = pkg_resources.resource_filename(__name__, 'weights/{}_final.pt'.format(args.model))
    state = torch.load(weight_path)
    net = NNmodel(**vars(args))
    net.load_state_dict(state["net_state_dict"])
    net.to(args.device)
    net.eval()    
    params_prompt = np.hstack((params, np.tile(args.initial_value, (params.shape[0], 1))))
    nn_sample_datas = nn_sample(params_prompt,net,args=args,param_indexs=range(0,params.shape[0],1),bs=args.batch_size,true_param=False)
    return nn_sample_datas

def pre_training(config_pretrain):
    """Pre-train the neural network model for state transition network.

    Args:
        config_pretrain (Namespace): Configuration object containing training parameters.

    Returns:
        dict: A dictionary containing training loss means and the final model weight path.
    """
    args = config_pretrain
    dataset_path = args.train_dataset_path
    args.out_dir = args.result_dir
    args.epochs =  args.epochs_pretrain
    device = args.device
    net = NNmodel(**vars(args))
    net.to(device)
    optimizer = net.configure_optimizers(
        args.weight_decay,
        args.lr,
        (args.beta1, args.beta2),
        device_type=args.device
    )
    if args.load_weight:
        model_state = torch.load(args.weight_file_path)
        net.load_state_dict(model_state["net_state_dict"])
        optimizer.load_state_dict(model_state["optimizer_state_dict"])

    raw_model = net
    ensure_dir(args.out_dir +"weights/" )
    ensure_dir(args.out_dir + "train_state/")
    ensure_dir(args.out_dir + "train_loss/")

    loss_means = []
    loss_stds = []

    prompt_prob_datasets = load_promt_prob(args,file_path = dataset_path)
    simulation_num = sum(prompt_prob_datasets[0][1].values())

    for epoch in range(args.start_epoch, args.epochs):
        data_size = 100
        batch_size = args.batch_size
        lr = get_lr(epoch) if args.decay_lr else args.lr
        set_learning_rate(optimizer, lr)
        loss_std_collection = [] 
        optimizer.zero_grad()  # accumulate weights for several prompts with small batch
        random.shuffle(prompt_prob_datasets)

        for i in range(data_size):
            prompt_prob =  prompt_prob_datasets[random.randint(0,len(prompt_prob_datasets)-2)]
            with torch.no_grad():
                prompt = prompt_prob[0]
                if torch.isinf(torch.tensor(prompt)).any():
                    continue

                if args.weighted_sample:
                    sample_weight = np.array(list(prompt_prob[1].values()))
                    sample_weight = sample_weight/sum(sample_weight)
                    samples_str = np.random.choice(list(prompt_prob[1].keys()),args.batch_size,p=sample_weight)
                else:
                    samples_str = np.random.choice(list(prompt_prob[1].keys()),batch_size)
                samples_int = np.array([el.split("_") for el in samples_str ]).astype(int)

                # Truncate samples using the maximum value
                for c_index in range(len(args.constrains)):
                    samples_int[samples_int[:,c_index] >= args.constrains[c_index],c_index] = args.constrains[c_index]-1

                prompt = np.repeat(np.array(prompt)[np.newaxis, :], batch_size, axis=0)
                pro_sam = np.hstack([prompt,samples_int])
                pro_sam = torch.as_tensor(pro_sam, dtype=args.default_dtype_torch, device=device)
                y_label = torch.as_tensor(samples_int, dtype=torch.long, device=device)
            loss_cross = net(pro_sam,y_label)
            loss_std_collection.append(loss_cross.detach().cpu().numpy())
            optimizer.zero_grad(set_to_none=True)
            loss_cross.backward()
        if args.clip_grad:
            nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)

        optimizer.step()
        ls = np.mean(loss_std_collection)
        loss_means.append(ls)
        print(
            f"|epoch {epoch}| cross loss: {ls:.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}")

        if epoch % 100 == 0:
            with torch.no_grad():
                weight_path = f"{args.out_dir}/weights/{args.model}_epoch{epoch}.pt"
                torch.save({
                    # "net_state_dict": net.state_dict(),
                    "net_state_dict": raw_model.state_dict(),
                    "dtype": args.dtype,
                    "optimizer_state_dict": optimizer.state_dict(),
                }, weight_path)
                loss_data_path = f"{args.out_dir}/train_loss/{args.model}_epoch{epoch}.npz"
                np.savez(
                    loss_data_path,
                    loss_mean=np.array(loss_means),
                    loss_std=np.array(loss_stds)
                )
                figure_path = f"{args.out_dir}/train_loss/{args.model}_epoch{epoch}.png"

    weight_path = f"{args.out_dir}/weights/{args.model}_pretrain_final.pt"
    torch.save({
        "net_state_dict": raw_model.state_dict(),
        "dtype": args.dtype,
        "optimizer_state_dict": optimizer.state_dict(),
    }, weight_path)

    result_dict = {}
    result_dict["loss_means"] = loss_means
    result_dict["weight_nn_pretrain_path"] = weight_path
    return result_dict

def fine_tune_training(config_fine_tune):
    """Fine-tune the neural network model using a specified training configuration.

    Args:
        config_fine_tune (Namespace): Configuration object containing hyperparameters, 
            dataset paths, training settings, and model specifications.

    Raises:
        ValueError: If an unknown loss type is specified in `args.loss_type`.

    Returns:
        dict: A dictionary containing training results, including loss means and the path 
            to the fine-tuned model weights.
    """  
    # start iterating
    args = config_fine_tune
    dataset_path = args.train_dataset_path
    args.load_weight = True
    args.out_dir = args.result_dir
    args.epochs =  args.epochs_fine_tune
    weight_file_path = args.weight_nn_pretrain_path
    args.sample = True 
    args.ssa_sample = True
    args.sample_num = 500
    args.ssa_sample_num = 500
    ensure_dir(args.out_dir +"weights/" )
    ensure_dir(args.out_dir + "train_state/")
    ensure_dir(args.out_dir + "train_loss/")
    device = args.device
    net = NNmodel_FT(**vars(args))
    net.to(device)
    optimizer = net.configure_optimizers(
        args.weight_decay,
        args.lr,
        (args.beta1, args.beta2),
        device_type=args.device
    )
    if args.load_weight: 
        model_state = torch.load(weight_file_path)
        net.load_state_dict(model_state["net_state_dict"])
        optimizer.load_state_dict(model_state["optimizer_state_dict"])

    loss_means = []
    loss_stds = []
    eds = []
    kls = []

    # load datasets
    prompt_prob_datasets = load_promt_prob(args,file_path = dataset_path)
    simulation_num = sum(prompt_prob_datasets[0][1].values())
    for epoch in range(args.start_epoch, args.epochs):
        lr = get_lr(epoch,learning_rate = 8e-4,lr_decay_iters= 10000,min_lr = 6e-5) if args.decay_lr else args.lr
        set_learning_rate(optimizer, lr)
        loss_mean_collection,loss_std_collection = [],[]   
        kl_distance_collection,euclidean_distance_collection = [],[]
        optimizer.zero_grad()  # accumulate weights for several prompts with small batch
        random.shuffle(prompt_prob_datasets)
        data_size = 100
        for i in range(data_size):
            with torch.no_grad():
                prompt_prob = prompt_prob_datasets[i]

            if args.sample:
                with torch.no_grad():
                    prompt = torch.as_tensor(prompt_prob[0], dtype=args.default_dtype_torch, device=args.device)
                    if torch.isinf(prompt).any():
                        continue
                    prompt = prompt.repeat(args.batch_size, 1)
                    pro_sam, samples = net.sample(prompt)  # pro_sam: prompt + samples as one complete sentence
                    samples = samples.detach()
                    log_Tp_t = [prompt_prob[1].get('_'.join(list(map(str,sample.int().tolist()))), args.epsilon2) for sample in samples]
                    log_Tp_t = np.log(np.array(log_Tp_t)/simulation_num)
                    log_Tp_t = torch.as_tensor(log_Tp_t, dtype=args.default_dtype_torch, device=args.device)
            
            if args.ssa_sample and  args.sample:
                pro_sam_nn=pro_sam
                log_Tp_t_nn = log_Tp_t

            if args.ssa_sample:
                prompt = prompt_prob[0]
                samples_str = np.random.choice(list(prompt_prob[1].keys()),args.batch_size)
                samples_int = np.array([el.split("_") for el in samples_str ]).astype(int)
                for c_index in range(len(args.constrains)):
                    samples_int[samples_int[:,c_index] >= args.constrains[c_index],c_index] = args.constrains[c_index]-1

                prompt = np.repeat(np.array(prompt)[np.newaxis, :], args.batch_size, axis=0)
                pro_sam = np.hstack([prompt,samples_int])
                pro_sam = torch.as_tensor(pro_sam, dtype=args.default_dtype_torch, device=args.device)    
                log_Tp_t = [prompt_prob[1].get(sample, args.epsilon) for sample in samples_str]
                log_Tp_t = np.log(np.array(log_Tp_t)/simulation_num)
                log_Tp_t = torch.as_tensor(log_Tp_t, dtype=args.default_dtype_torch, device=args.device)
            
            if args.ssa_sample and  args.sample:
                pro_sam = torch.cat([pro_sam_nn[:args.sample_num,:],pro_sam[:args.ssa_sample_num,:]])
                log_Tp_t = torch.cat([log_Tp_t_nn[:args.sample_num ],log_Tp_t[:args.ssa_sample_num]])                   
            log_prob = net.log_joint_prob(pro_sam,unobserved_number = 0)

            with torch.no_grad():
                prob = torch.exp(log_prob.detach())
                r_prob = torch.exp(log_Tp_t.detach())
                r_prob = r_prob / r_prob.sum()
                r_prob = (r_prob * prob.sum()).detach()
                loss = log_prob - log_Tp_t.detach()
                loss_l2 = prob - r_prob
                loss_he = -torch.sqrt(prob * r_prob)
            assert not log_Tp_t.requires_grad

            if args.loss_type == 'kl':
                loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)

            elif args.loss_type == 'klreweight':
                loss3 = prob * loss / prob.mean()
                loss_reinforce = torch.mean((loss3 - loss3.mean()) * log_prob)
            elif args.loss_type == 'l2':
                loss_reinforce = torch.mean((loss_l2 - loss_l2.mean()) * log_prob)
            elif args.loss_type == "crossEntropy":
                loss_reinforce = -torch.sum(r_prob * log_prob)
            else:
                raise ValueError('Unknown loss type: {}'.format(args.loss_type))

            loss_reinforce.backward()
            loss_std = loss.std()
            loss_mean = loss.mean()
            loss_mean_collection.append(loss_mean.detach().cpu().numpy())
            loss_std_collection.append(loss_std.detach().cpu().numpy())
            f = (prob - r_prob) ** 2
            euclidean_distance = torch.sqrt(torch.sum(f))
            kl_distance = torch.nn.functional.kl_div(
                log_prob,
                r_prob,
                None,
                None,
                'sum'
            )
            euclidean_distance_collection.append(euclidean_distance.detach().cpu().numpy())
            kl_distance_collection.append(kl_distance.detach().cpu().numpy())

        if args.clip_grad:
            nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
        optimizer.step()
        lm = np.mean(loss_mean_collection)
        ls = np.mean(loss_std_collection)
        ed = np.mean(euclidean_distance_collection)
        kl = np.mean(kl_distance_collection)
        loss_means.append(lm)
        loss_stds.append(ls)
        eds.append(ed)
        kls.append(kl)
        print(
        f"|epoch {epoch}|mean loss: {lm:.4f}, std loss: {ls:.4f}, ed: {ed:.4f}, kd: {kl:.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}")
        if epoch % 100 == 0:
            with torch.no_grad():
                path = f"{args.out_dir}/weights/fine_tune_{args.model}_epoch{epoch}.pt"
                torch.save({
                    "net_state_dict": net.state_dict(),
                    "dtype": args.dtype,
                    "optimizer_state_dict": optimizer.state_dict(),
                }, path)
                path1 = f"{args.out_dir}/train_loss/fine_tune_{args.model}_epoch{epoch}.npz"
                np.savez(
                    path1,
                    loss_mean=np.array(loss_means),
                    loss_std=np.array(loss_stds),
                    ed=np.array(eds),
                    kl=np.array(kls),
                )

    weight_path = f"{args.out_dir}/weights/fine_tune_{args.model}_final.pt"
    torch.save({
        "net_state_dict": net.state_dict(),
        "dtype": args.dtype,
        "optimizer_state_dict": optimizer.state_dict(),
    }, weight_path)
    result_dict = {}
    result_dict["loss_means"] = loss_means
    result_dict["weight_nn_fine_tune_path"] = weight_path
    return result_dict

def solving_STN(model_name,params=[],initial_value=[],args=""):
    """Solve the state transition network using a trained neural network model.

    Args:
        model_name (str): The name of the model to be used.
        params (list, optional): A list of parameters for the model. Defaults to [].
        initial_value (list, optional): A list of initial values for the system. Defaults to [].
        args (Namespace, optional): Additional arguments containing model configurations, 
                                    including device, batch size, and initial values. Defaults to "".

    Returns:
        np.ndarray: Sampled data generated by the neural network based on the provided parameters.
    """  
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    CUDA_LAUNCH_BLOCKING=1
    weight_path = pkg_resources.resource_filename(__name__, 'weights/{}_final.pt'.format(args.model))
    state = torch.load(weight_path)
    net = NNmodel(**vars(args))
    net.load_state_dict(state["net_state_dict"])
    net.to(args.device)
    net.eval()    
    params_prompt = np.hstack((params, np.tile(args.initial_value, (params.shape[0], 1))))
    nn_sample_datas = nn_sample(params_prompt,net,args=args,param_indexs=range(0,params.shape[0],1),bs=args.batch_size,true_param=False)
    return nn_sample_datas
