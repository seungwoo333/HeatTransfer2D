import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
from model import BranchNet_RGNN
from dataloader import adj_mat, dataset_train, dataset_test, base_path, data_path, SolutionDataset_RGNN, GetAdjMat, Nt

result_path = base_path + '/results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(data_path + 'dataset_train.pickle', 'rb') as f:
    dataset_train = pickle.load(f)
with open(data_path + 'dataset_test.pickle', 'rb') as f:
    dataset_test = pickle.load(f)


net = BranchNet_RGNN(adj=adj_mat, num_RNN=1, hidden_size=Nt, in_feature=Nt, out_feature=Nt).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_idx = range(len(dataset_train))
batch_size = 5
train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size = batch_size, shuffle=False, drop_last=True)

print_every = 1

num_epoch = 20
train_loss_ary = []
test_loss_ary = []
relaxation_ary = []

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
    relaxation_ary.append(relaxation_curent)
    
    if (epoch+1) % 10 == 0:
        torch.save(net, result_path + f'epoch{epoch+1}.pt')


with open(result_path + 'train_loss.pickle', 'wb') as f:
    pickle.dump(train_loss_ary, f)
with open(result_path + 'test_loss.pickle', 'wb') as f:
    pickle.dump(test_loss_ary, f)
with open(result_path + 'relaxation.pickle', 'wb') as f:
    pickle.dump(relaxation_ary, f)

plt.figure()
plt.plot(train_loss_ary, label='train')
plt.plot(test_loss_ary, label='test')
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
plt.legend()
plt.savefig(base_path + '/fig/loss.png')

plt.figure()
plt.plot(relaxation_ary)
plt.xlabel('Epoch')
plt.ylabel('relaxation')
plt.savefig(base_path + '/fig/relaxation.png')

plt.show()