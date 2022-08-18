import torch
from torch.utils.data import Dataset
from numpy import genfromtxt, exp
import multiprocessing
import pickle

"""
Load dataset

Nt: # of node
nodeConnectivity (Nt, 3) [i, :] -> node number in element i
nodeCoordinate (Nt, 2) [i, :] -> x, y coodinate of node i
adjacency matrix from ndode coordinate and connectivity

dataset_train
dataset_test
input: nodal temperature [batch, Nt, sequence]

"""
base_path = '.'
data_path = base_path + '/matlab/data/'

nodeConnectivity = genfromtxt(data_path + 'nodeConnectivity.csv', delimiter=',')
nodeCoordinate = genfromtxt(data_path + 'nodeCoordinate.csv', delimiter=',')
Nt = len(nodeCoordinate)

def kernel(x1, x2):
    d = ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**0.5
    return float(exp(-d))

def GetAdjMat():
    Nt = len(nodeCoordinate)

    adj = torch.zeros(Nt, Nt)

    for element in torch.from_numpy(nodeConnectivity).type(torch.long):
        for n1 in element:
            for n2 in element:
                adj[n1-1, n2-1] = kernel(nodeCoordinate[n1-1], nodeCoordinate[n2-1])

    adj = adj + torch.diag(torch.count_nonzero(adj, dim=1)-1)
    return adj


#functions originally used for dump csv data to pickle (list of tensors)
def GetDP(sol, mode):
    input_filename = f'temperature_sol{sol}.csv'
    label_filemane = f'diffusion_sol{sol}.csv'

    inputs = genfromtxt(data_path + input_filename, delimiter=',')
    inputs = torch.tensor(inputs).type(torch.float32)

    if mode == 'temperature_RNN':
        return inputs[:-1, :], inputs[1:, :]

    labels = genfromtxt(data_path + label_filemane, delimiter=',')
    labels = torch.tensor(labels).type(torch.float32)

    if mode == 'diffusion_RNN':
        inputs_temp = torch.unsqueeze(inputs_temp, dim=1)
        return inputs_temp[:-1, :, :], labels

    n_sample = inputs.size(0)
    inputs = torch.repeat_interleave(inputs, Nt*torch.ones(n_sample).type(torch.long), dim=0)
    coordinates = torch.tensor(nodeCoordinate).type(torch.float32)
    coordinates = coordinates.repeat(n_sample, 1)
    inputs = torch.cat([inputs, coordinates], dim=1)

    inputs = inputs.reshape(n_sample, Nt, Nt+2)

    if mode == 'temperature':
        return inputs[:-1, :, :], inputs_temp[1:, :]

    elif mode == 'diffusion':
        return inputs[:-1, :, :], labels

def RNN_dataloader(sol):
    return GetDP(sol, 'temperature_RNN')

def Dump_multi(sol_idx, phase):
    pool_obj = multiprocessing.Pool(4)
    res = pool_obj.map(RNN_dataloader, sol_idx)
    with open(data_path + f'dataset_{phase}.pickle', 'wb') as f:
        pickle.dump(res, f)


class SolutionDataset_RGNN(Dataset):
    def __init__(self, sol_idx, dataset):
        self.sol_idx = sol_idx
        self.dataset=dataset

    def __len__(self):
        return len(self.sol_idx)

    def __getitem__(self, idx):
        return self.dataset[idx]


adj_mat = GetAdjMat()

train_idx = range(45)
test_idx = range(45, 50)
#Dump_multi(train_idx, 'train')
#Dump_multi(test_idx, 'test')

with open(data_path + 'dataset_train.pickle', 'rb') as f:
    dataset_train = pickle.load(f)
with open(data_path + 'dataset_test.pickle', 'rb') as f:
    dataset_test = pickle.load(f)

dataset_train=SolutionDataset_RGNN(train_idx, dataset_train)
dataset_test=SolutionDataset_RGNN(test_idx, dataset_test)