# MODES
The Repository can be used with an uv env by syncing the environment 

The training and testing can be done with the scripts main.py and test.py. 
The shell scripts are used for training on a GPU-cluster.

fusion.py contains the fusion model and the extension with the MI estimation.
The data structures and the dataset methods can be found in data.py. 
The dataloader needs a custom collate_fn, which is contained in the utils.py. 
The code for evaluating the factors against the representations is contained in the evaluation.py 


