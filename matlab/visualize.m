clear all; close all;

T = readtable('./eval/solution.csv', 'NumHeaderLines', 1);
T = table2array(T(:, 2:end));
nodeCoordinate = readmatrix('./data/nodeCoordinate.csv');


initial = T(1, :);
final = T(end, :);

% Copper
k = 398; % Thermal conductivity, W/(m*C)
rho = 8960; % Density, kg/m^3
cp = 385; % Specific heat, W*s/(kg*C)

[model] = ThermalModel(k, rho, cp);

figure
pdeplot(model, 'XYData', initial', 'ColorMap', 'hot');
title('initial temperature dist')
axis equal;

figure
pdeplot(model, 'XYData', final', 'ColorMap', 'hot')
title('final temperature dist')
axis equal;

figure
profile = [];
[num_timestep, ~] = size(T);
for timestep = 1:num_timestep
    F = scatteredInterpolant(nodeCoordinate,T(timestep, :)');
    profile = [profile, F(0, 0.4)];
end
plot(linspace(1, 10, num_timestep), profile);