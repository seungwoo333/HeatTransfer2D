clear all; close all;

%% Define model and iter
% Copper
k = 398; % Thermal conductivity, W/(m*C)
rho = 8960; % Density, kg/m^3
cp = 385; % Specific heat, W*s/(kg*C)

[model] = ThermalModel(k, rho, cp);
num_solution = 50;
plot_every = 10;

mesh = model.Mesh;
nodeConnectivity = mesh.Elements';
nodeCoordinate = mesh.Nodes';
[num_nodes, ~] = size(nodeCoordinate);


save = false;
save_from = 0;

save_plot = true;


base = "./data/"

if save
    writematrix(nodeConnectivity, base + "/nodeConnectivity.csv")
    writematrix(nodeCoordinate,  base + "/nodeCoordinate.csv"')
end


%% main loop
for i = 0:num_solution-1
    % Set initial, boundary condition
    [T0] =GenerateRandomIC(model);
    thermalIC(model,T0);
    Tb = @(location, state) (T0(location));
    thermalBC(model,'Edge',[1, 2, 3, 4, 5],'HeatFlux',0);
    
    
    % Solve
    dt=0.1;
    tlist=0:dt:10;
    result = solve(model, tlist);
    
    time_mask = 12:numel(tlist);
    T = result.Temperature(:, time_mask);
    D = (T(:, 2:end) - T(:, 1:end-1)) / dt;
    
    
    fprintf('saving [%d / %d] solution', i+1, num_solution);
    if save
        writematrix(T',  sprintf(base + "temperature_sol%d.csv", save_from + i));
        writematrix(D',  sprintf(base + "diffusion_sol%d.csv", save_from + i));
    end
    
            
    % Plot
    if rem(i, plot_every) == 0
        figure
        pdeplot(model, 'XYData', T(:, 1), 'ColorMap', 'hot');
        title('initial temperature dist')
        axis equal;
        if save_plot
            saveas(gcf, sprintf(base + 'fig/initial_sol%d.png', save_from+i))
        end
        
        figure
        pdeplot(model, 'XYData', T(:, end), 'ColorMap', 'hot')
        title('final temperature dist')
        axis equal;
        if save_plot
            saveas(gcf, sprintf(base + 'fig/final_sol%d.png', save_from+i))
        end
        
        figure
        Tcenter = interpolateTemperature(result,[0; 0.4], time_mask);
        plot(tlist(time_mask), Tcenter);
        if save_plot
            saveas(gcf, sprintf(base + 'fig/profile_x00y04_sol%d.png', save_from+i))
        end
    end
    close all;
    
    
end