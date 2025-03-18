import numpy as np
import torch

def get_isc_model_config():
    from dyngpt.nnmodel.hyperparameters import args
    # --------------------------------- model architecture -------------------------------------#
    args.model = 'isc'
    args.num_species = 10  # Species number
    args.num_reactions = 3  # reaction number
    args.num_init = 10 # Species number of init

    # Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation.
    args.state_upper_bound = int(113)
    args.constrains = np.array([113,62,42,33,28,24,22,19,18,90], dtype=int)
    # --------------------------------- train NNnet -------------------------------------#
    args.bits = 1
    args.variable_dimension = 64  # dimensionality of the variable matrix including prompt and states
    args.embedding_dimension = 64  # transformer emb_dim
    args.feed_forward_dimension = 1024  # transformer ff_dim
    args.num_encoder_layers = 1  # transformer n_layer
    # args.num_encoder_layers = 1  # transformer n_layer
    args.n_head = 8  # transformer n_head
    args.block_size = 128  # maximum input length for dyngpt
    args.lr = 0.001  # initial learning rate
    args.batch_size = 1000
    args.bias = False  # False for training dyngpt
    args.dropout_rate = 0.0  # for dyngpt
    args.weight_decay = 1e-1  # for configure dyngpt optimizer
    args.beta1 = 0.9  # for configure dyngpt optimizer
    args.beta2 = 0.999  # for configure dyngpt optimizer
    args.decay_lr = True  # whether to decay the learning rate
    args.epochs = 10000  # usually should be 5000-10000 epochs for convergent training
    args.start_epoch = 0  # changed when loading pretrain dyngpt state file
    args.last_epoch = 4999  # specify last epoch for loading dyngpt pretrain state file
    return args

