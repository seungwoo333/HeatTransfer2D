close all; clear all;

k = 398; % Thermal conductivity, W/(m*C)
rho = 8960; % Density, kg/m^3
cp = 385; % Specific heat, W*s/(kg*C)

[model] = ThermalModel(k, rho, cp);
num_solution = 200;
plot_every = 50;

mesh = model.Mesh;
nodeConnectivity = mesh.Elements';
nodeCoordinate = mesh.Nodes';
[num_nodes, ~] = size(nodeCoordinate);

writematrix(nodeConnectivity, './data_eval/nodeConnectivity.csv')
writematrix(nodeCoordinate, './data_eval/nodeCoordinate.csv')

[T0] =GenerateRandomIC(model);
thermalIC(model,T0);
thermalBC(model,'Edge',[1, 2, 3, 4, 5],'HeatFlux',0);


% Solve
dt=0.01;
tlist=0:dt:10;
result = solve(model, tlist);
T = result.Temperature;

D = (T(:, 2:end) - T(:, 1:end-1)) / dt;


figure
pdeplot(model, 'XYData', T(:, 1), 'ColorMap', 'hot');
title('initial temperature dist')
axis equal;

figure
pdeplot(model, 'XYData', T(:, end), 'ColorMap', 'hot')
title('final temperature dist')
axis equal;

figure
Tcenter = interpolateTemperature(result,[0; 0.4],1:numel(tlist));
plot(tlist, Tcenter);


writematrix(T(:, 1:end-1),  sprintf('./data_eval/inputs_branch_solution_eval.csv'));
writematrix(D,  sprintf('./data_eval/labels_branch_solution_eval.csv'));

T_exact = [T(:, 1)];
for step = 1:1000
    dT = D(:, step);
    T_exact = [T_exact, T_exact(:, end) + dT * dt];
end