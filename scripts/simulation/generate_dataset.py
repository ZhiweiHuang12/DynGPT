import json
import numpy as np 
import random
import pandas as pd 
def load_json(file_path=""):
    with open(file_path, 'r') as f:
        s1 = json.load(f)
    return s1
def save_json(data,file_path):
    with open(file_path, 'w') as f:
        json.dump(data,f)

def generate_file2(param,data, data_size,init,path_flag=True):
    if path_flag:
        parameters_df = pd.read_csv(param)
        prob_file = load_json(data)
    else:
        parameters_df = param
        prob_file = data
    if model_type == "arl":
        parameters_df.iloc[2,:] = parameters_df.iloc[2,:]+2
    parameters_df_log = parameters_df.apply(np.log)
    parameters = parameters_df_log.transpose().values
    t = np.zeros((data_size,1))
    inits = np.tile(init, (data_size,1))
    parameters_init_t = np.hstack([parameters,inits,t])
    dataset =[[list(parameters_init_t[i]),prob_file[i]] for i in range(data_size)]
    return dataset

model_type = "autoRegulationModel"
root_path = "your/simulate/data_path/"
single_flag = True
if model_type=="NonMarkovianRNAsplicingModel":
    init = [ 1,0,0 ]
    single_flag = False
    data_train_li = [[3000,545],[2900,557],[3010,564],[3100,575],[1030,139],[910,148],[920,131],[930,142],[940,131],[950,133],[960,137],[970,127],[980,130]]
    data_train_li = [[10000,198],[910,148],[920,131],[930,142],[940,131],[950,133],[960,137],[970,127],[980,130]]

    ds_valid,valid_max_val = 1000,180
    if single_flag:
        ds_valid,valid_max_val = 1000,180
    else:
        data_valid_li = [[68,25],[67,47],[66,35],[65,75],[64,53],[63,39],[62,53],[61,34],]

elif model_type=="centralDogmaModel":
    init = [1,0,0]
    data_train_li = [[40000,577],[10000,574],[4000,135],[3000,552],[5000,555]]
    single_flag = False
    if single_flag:
        ds_valid,valid_max_val = 1000,499
    else:
        data_valid_li = [[1000,499],[1100,500],[104,95],[103,315],[105,329]]

elif model_type=="toggle_switch":
    init = [ 1,1,0,0 ]
    single_flag = False
    data_train_li = [[5000,352],[25000,344],[20000,368]]
    if single_flag:
        ds_valid,valid_max_val = 500,339
    else:
        data_valid_li = [[500,339],[1000,311],[600,367],[200,282]]

elif model_type=="epidemicModel":
    init = [ 1,1,0 ]
    single_flag = False
    # ds_train,ds_valid,train_max_val,valid_max_val = 16000,1600,228,229
    data_train_li = [[40000,242]]
    data_train_li = [[100000,200]]
    if single_flag:
        ds_valid,valid_max_val = 1000,210
        ds_valid,valid_max_val = 1000,176
    else:
        data_valid_li = [[1000,176]]

elif model_type=="autoFeedbackLoop":
    init = [0,1]
    single_flag = False
    ds_train,ds_valid,train_max_val,valid_max_val = 20000,1000,152,144,
    data_train_li = [[20000,152],[11000,144],[10000,151],[5500,28],[5100,49]]
    if single_flag:
        ds_valid,valid_max_val = 1600,229
    else:
        data_valid_li = [[1000,144],[1100,142]]
elif model_type=="autoRegulationModel":
    init = [0,1]
    single_flag = False
    data_train_li = [[20000,1054]]
    data_train_li = [[100000,430]]
    data_train_li = [[60000,435]]

    if single_flag:
        ds_valid,valid_max_val = 1600,229
    else:
        data_valid_li = [[2000,353]]

elif model_type=="MarkovianRNAsplicingModel": 
    init = [1,0,0]
    ds_train,ds_valid,train_max_val,valid_max_val = 30000,1000,1386,622,
    data_train_li = [[31000,250],[11000,428],[12000,454]]
    single_flag = False
    if single_flag:
        ds_valid,valid_max_val = 1600,229
    else:
        data_valid_li = [[1100,196],[900,304],[1200,389]]


train_save_path = root_path+"{}/{}_train_stable.json".format(model_type,model_type,)
valid_save_path = root_path+"{}/{}_valid_stable.json".format(model_type,model_type,)

param_file_li = []
prob_file_li = []
for i in range(len(data_train_li)):
    ds_train,train_max_val = data_train_li[i]
    train_para_path = root_path + "{}/train_{}_stable_parameter_{}.csv".format(model_type,model_type,ds_train)
    train_prob_path = root_path + "{}/train_{}_stable_{}_{}.json".format(model_type,model_type,train_max_val,ds_train)
    parameters_df = pd.read_csv(train_para_path)
    prob_file = load_json(train_prob_path)
    param_file_li.append(parameters_df.values)
    prob_file_li.extend(prob_file)
param_data = np.hstack(param_file_li)
train_dataset = generate_file2(pd.DataFrame(param_data),prob_file_li,len(prob_file_li),init = init,path_flag=False)

if single_flag:
    valid_para_path = root_path + "{}/valid_{}_stable_parameter_{}.csv".format(model_type,model_type,ds_valid)
    valid_prob_path = root_path + "{}/valid_{}_stable_{}_{}.json".format(model_type,model_type,valid_max_val,ds_valid)
    valid_dataset = generate_file2(valid_para_path,valid_prob_path,ds_valid,init = init)
else:
    valid_param_file_li = []
    valid_prob_file_li = []
    for i in range(len(data_valid_li)):
        ds_valid,valid_max_val = data_valid_li[i]
        valid_para_path = root_path + "{}/valid_{}_stable_parameter_{}.csv".format(model_type,model_type,ds_valid)
        valid_prob_path = root_path + "{}/valid_{}_stable_{}_{}.json".format(model_type,model_type,valid_max_val,ds_valid)
        parameters_df = pd.read_csv(valid_para_path)
        prob_file = load_json(valid_prob_path)
        valid_param_file_li.append(parameters_df.values)
        valid_prob_file_li.extend(prob_file)
    valid_param_data = np.hstack(valid_param_file_li)
    valid_dataset = generate_file2(pd.DataFrame(valid_param_data),valid_prob_file_li,len(valid_prob_file_li),init = init,path_flag=False)

save_json(train_dataset, train_save_path)
save_json(valid_dataset, valid_save_path)


