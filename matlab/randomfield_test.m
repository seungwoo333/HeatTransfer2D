clear all; close all;

%% 1D
corr.name = 'gauss';
corr.c0 = 1;
corr.sigma = 1;
mesh = linspace(-1,1,101)';
data.x = [-1; 1]; data.fx = [0; 0];    %boundary values F(-1)=0, F(1)=-1
[F,KL] = randomfield(corr, mesh, 'nsamples', 5, 'data', data);

figure()
for i=1:5
    plot(F(:, i))
    hold on
end
           
%% 2D
corr.name = 'exp';
corr.c0 = [1]; % anisotropic correlation
corr.sigma = 1;

x = linspace(0,1,21);
[X,Y] = meshgrid(x,x); mesh = [X(:) Y(:)]; % 2-D mesh

[F,KL] = randomfield(corr,mesh);

figure()
surf(X,Y,reshape(F,21,21)); view(2); colorbar;

T0 = @(location) interp2(X, Y, reshape(F,21,21), [location.x], [location.y], 'cubic');
thermalModel = createpde('thermal', 'transient');


T1 = [];
T = [];
x = linspace(0, 1, 51);
[X,Y] = meshgrid(x,x); mesh = [X(:) Y(:)]; % 2-D mesh
for x_cor=x
    for y_cor=x
        location.x = x_cor;
        location.y = y_cor;
        T1 = [T1 T0(location)];
    end
    T = [T; T1];
    T1=[];
end
figure()
surf(X, Y, T'); view(2);

%% 2D with geometry and mesh
plate = 'Fillets';
ratio = 0.5;
Hmax = 1e-3;
Hmin = 0.01e-3;
[nodeCoordinate, nodeConnectivity, model] = generateModel(plate, ratio, Hmax, Hmin);
pdemesh(model)

corr.name = 'exp';
corr.c0 = [0.1, 0.1]; % anisotropic correlation
corr.sigma = 1;
mesh = nodeCoordinate;
[F,KL] = randomfield(corr,mesh);

figure()
pdeplot(model,"XYData",F);
axis equal;

figure()
nodes_axis = nodeCoordinate(:, 2) < 1e-5;
val_y0 = F(nodes_axis, 1);
plot(nodeCoordinate(nodes_axis, 1), val_y0);

%% Transient thermal model with random initial temperature field

% Geometry
L = 1;
rho = 0.25;
t = pi/24:pi/24:2*pi;
pgon = polyshape({[-L/2, -L/2, L/2, L/2], [rho*cos(t)] }, {[-L/2, L/2, L/2, -L/2], [rho*sin(t)]});

% Mesh
tr = triangulation(pgon);
tnodes = tr.Points';
telements = tr.ConnectivityList';
model = createpde('thermal','transient');
geometryFromMesh(model,tnodes,telements);
Hmax = 0.03;
Hmin = 0.01;
generateMesh(model, 'Hmax', Hmax, 'Hmin', Hmin, 'GeometricOrder', 'linear');
figure
pdemesh(model);

% Model setup
k = 398; % Thermal conductivity, W/(m*C)
rho =8960; % Density, kg/m^3
cp =385; % Specific heat, W*s/(kg*C)
thermalProperties(model,'ThermalConductivity',k, 'MassDensity',rho, 'SpecificHeat',cp);

mesh = model.Mesh;
nodeConnectivity = mesh.Elements';
nodeCoordinate = mesh.Nodes';
corr.name = 'exp';
corr.c0 = [0.1, 0.1]; % anisotropic correlation
corr.sigma = 1;
mesh = nodeCoordinate;
[F,KL] = randomfield(corr,mesh);

figure
pdegplot(model,'EdgeLabels','on')
axis equal
F_interp = scatteredInterpolant(nodeCoordinate(:, 1), nodeCoordinate(:, 2), F, 'natural');
T0 = @(location) 10 * F_interp(location.x, location.y);
thermalIC(model,T0);

thermalBC(model,'Edge',[1, 2, 3, 4, 5],'HeatFlux',0);

% Solve
tlist=0:0.1:1000;
result = solve(model, tlist);
T = result.Temperature;
qx = -k * result.XGradients;

%Post
figure
pdeplot(model, 'XYData', T(:, 1));
axis equal;
figure
pdeplot(model, 'XYData', T(:, end))
axis equal;
figure
pdeplot(model, 'XYData', qx(:, 1))
axis equal;

Tcenter = interpolateTemperature(result,[0; 0.4],1:numel(tlist));
figure
plot(tlist, Tcenter);