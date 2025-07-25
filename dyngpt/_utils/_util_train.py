import math
import random
import torch

def get_lr(it,learning_rate = 1e-3,warmup_iters = 20,lr_decay_iters= 1000,min_lr = 6e-4):
    # adamw optimizer
    # learning_rate = 8e-4 # max learning rate
    # warmup_iters = 20  # how many steps to warm up for
    # lr_decay_iters = 10000  # should be ~= max_iters per Chinchilla
    # min_lr = 6e-5 

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)




# def get_lr(it):
#     # adamw optimizer
#     learning_rate = 1e-3  # max learning rate
#     warmup_iters = 20  # how many steps to warm up for
#     lr_decay_iters = 1000  # should be ~= max_iters per Chinchilla
#     min_lr = 6e-4  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_iters:
#         return learning_rate * it / warmup_iters
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > lr_decay_iters:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
#     return min_lr + coeff * (learning_rate - min_lr)

def configure_optimizer(net, learning_rate):
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    num_params = int(sum([np.prod(p.shape) for p in params]))
    # adam = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
    adam = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999))

    return adam, params, num_params


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def load_promt_prob(args):
    file_path = args.data_dir+"{}_train_stable.json".format(args.model)
    with open(file_path, 'r') as f:
        s1 = json.load(f)
    return s1

