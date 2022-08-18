from numpy import genfromtxt, exp
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pickle
import multiprocessing

device = torch.device('cuda:0')

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
data_path = base_path + '/data/'

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

adj_mat = GetAdjMat().to(device)


#functions originally used for dump csv data to pickle (list of tensors)
def GetBatch(sol, mode):
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
    return GetBatch(sol, 'temperature_RNN')

def SaveBatches_multi(sol_idx, phase):
    pool_obj = multiprocessing.Pool(4)
    res = pool_obj.map(RNN_dataloader, sol_idx)
    with open(f'./dataset_5000/dataset_{phase}.pickle', 'wb') as f:
        pickle.dump(res, f)


class SolutionDataset_RGNN(Dataset):
    def __init__(self, sol_idx, dataset):
        self.sol_idx = sol_idx
        self.dataset=dataset

    def __len__(self):
        return len(self.sol_idx)

    def __getitem__(self, idx):
        return self.dataset[idx]


with open(f'./data/dataset_train.pickle', 'rb') as f:
    dataset_train = pickle.load(f)
with open(f'./data/dataset_test.pickle', 'rb') as f:
    dataset_test = pickle.load(f)

train_idx = range(4500)
test_idx = range(4500, 5000)
dataset_train=SolutionDataset_RGNN(train_idx, dataset_train)
dataset_test=SolutionDataset_RGNN(test_idx, dataset_test)


"""
Define model
"""
class BranchNet_RGNN(nn.Module):
    def __init__(self, adj, num_RNN, hidden_size, in_feature, out_feature):
        super().__init__()
        self.num_RNN = num_RNN
        self.hidden_size = hidden_size

        self.adj = nn.Parameter(adj, requires_grad=False)
        self.adj_weight = nn.Parameter(torch.randn_like(adj), requires_grad=True)
        self.RNN = nn.RNN(input_size = in_feature, hidden_size = hidden_size, num_layers = num_RNN, batch_first = True)
        self.fc = nn.Linear(hidden_size, out_feature, bias=True)
        self.relaxation = nn.Parameter(torch.tensor([0.001]), requires_grad=True)

    def forward(self, x, hidden):
        out = x @ (self.adj * self.adj_weight)
        out, hidden = self.RNN(out, hidden)
        out = self.fc(out)
        out = self.relaxation * out + (1-self.relaxation) * x
        return out, hidden, self.relaxation

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_RNN, batch_size, self.hidden_size)

    
net = BranchNet_RGNN(adj=adj_mat, num_RNN=1, hidden_size=Nt, in_feature=Nt, out_feature=Nt).to(device)


"""
main loop
"""
result_path = base_path + '/results/'

optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

batch_size = 100
train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size = batch_size, shuffle=False, num_workers=4, drop_last=True)

print_every = 5

num_epoch = 2000
train_loss_ary = []
test_loss_ary = []
relaxation_array = []

for epoch in range(num_epoch):
    print(f"-----Epoch {epoch + 1} / {num_epoch}-----")
    with torch.no_grad():
        test_loss_avg = 0
        batch_cnt = 0
        for batch in test_loader:
            hidden = net.init_hidden(batch_size).to(device)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            out, hidden, relaxation = net(inputs, hidden)
            loss = criterion(out, labels) + (1-relaxation) * 10**-6
            test_loss_avg += loss.item()
            batch_cnt += 1
        test_loss_avg /= batch_cnt
    train_loss_avg = 0
    batch_cnt = 0
    for i, batch in enumerate(train_loader):
        hidden = net.init_hidden(batch_size).to(device)
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out, hidden, relaxation = net(inputs, hidden)
        loss = criterion(out, labels) + (1-relaxation) * 10**-6
        loss.backward()
        optimizer.step()

        train_loss_avg += loss.item()
        batch_cnt += 1
        
        if (i+1) % print_every == 0:
            print(f"[{i+1} / {len(train_idx) // batch_size}] current loss: {loss.item()}", flush=True)

    train_loss_avg /= batch_cnt


    relaxation_curent = net.relaxation.item()
    print(f"Epoch {epoch + 1} train loss: {train_loss_avg} test loss: {test_loss_avg}", flush=True)
    print(f'current relaxation coefficient: {relaxation_curent}\n')

    train_loss_ary.append(train_loss_avg)
    test_loss_ary.append(test_loss_avg)
    relaxation_array.append(relaxation_curent)
    
    if (epoch+1) % 10 == 0:
        torch.save(net, result_path + f'epoch{epoch+1}.pt')


with open(result_path + 'train_loss.pickle', 'wb') as f:
    pickle.dump(train_loss_ary, f)
with open(result_path + 'test_loss.pickle', 'wb') as f:
    pickle.dump(test_loss_ary, f)
with open(result_path + 'relaxation.pickle', 'wb') as f:
    pickle.dump(relaxation_array, f)