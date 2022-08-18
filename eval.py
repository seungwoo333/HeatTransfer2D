import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import abstractmethod
from dataloader import base_path, nodeCoordinate, dataset_test
from model import BranchNet_RGNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Solver:
  def __init__(self, model, initial, timestep, dt=0.1):
    self.initial = initial

    self.T_current = initial

    self.model = model

    self.timestep = timestep
    self.dt = dt

    self.T_log = [self.T_current, ]

  @abstractmethod
  def step(self):
    pass

  @abstractmethod
  def solve(self):
    pass

  def plot_profile(self, node):
    temp_profile = []
    for temp in self.T_log:
        temp_profile.append(temp[node].item())

    plt.figure()
    plt.plot(temp_profile)
    print(max(temp_profile))
    plt.show()
    plt.savefig(base_path + "/fig/temp.png")

  def plot_contour(self, timestep):
    dist = self.T_log[timestep]

    plt.figure()
    plt.tricontour(nodeCoordinate[:, 0], nodeCoordinate[:, 1], dist.cpu())
    plt.show()
    plt.savefig(base_path + "/fig/temp.png")

  def export(self):
    T_np = np.array(list(map(lambda x: x.detach().cpu().numpy(), self.T_log)))
    T_df = pd.DataFrame(T_np)
    T_df.to_csv(base_path + '/matlab/eval/solution.csv')


class Solver_RGNN(Solver):
  def __init__(self, model, initial, timestep, dt):
    super().__init__(model, initial, timestep, dt)
    self.hidden = model.init_hidden(1).to(device)
    self.T_input = self.T_current.reshape(1, 1, -1)

  def step(self):
    with torch.no_grad():
      self.T_current, self.hidden, relaxation = self.model(self.T_input, self.hidden)
      self.T_current = self.T_current[0, -1, :].reshape(1, 1, -1)
    self.T_input = torch.cat([self.T_input, self.T_current], dim=1)
    self.T_log.append(torch.squeeze(self.T_current))

  def solve(self):
    for time in range(self.timestep):
      self.step()


input_eval, _ = dataset_test[-1]
initial = input_eval[0, :].to(device)
dt = 0.1
timestep = 90 - 1
epoch = 10
model_path = base_path + f'/results/epoch{epoch}.pt'
net = torch.load(model_path, map_location=device)
solver = Solver_RGNN(net, initial, timestep, dt)

solver.solve()
solver.export()
solver.plot_profile(459)