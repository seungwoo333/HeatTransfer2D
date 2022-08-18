function  [nodeCoordinate, nodeConnectivity, model] = generateModel(plate, ratio, Hmax, Hmin)
L = 56e-3;
D = 20e-3;
F = 500;
t = 1e-3;
width = 20e-3;
T = F / (width * t);

if strcmp(plate, "CircularHole")
    d = D/(2*ratio + 1);
    rho = ratio * d;
    t = 0:pi/24:pi;
    pgon = polyshape({[-L/2, -L/2, L/2, L/2, rho*cos(t)] }, {[0, D/2, D/2, 0, rho*sin(t)]});
elseif strcmp(plate, "SemicircularGrooves")
    d = D/(2*ratio + 1);
    rho = ratio * d;
    t = 0:pi/24:pi;
    pgon = polyshape({[-L/2, -L/2, -rho*cos(t), L/2, L/2]}, {[0, D/2, D/2-rho*sin(t), D/2, 0]});
elseif strcmp(plate, "Fillets")
    d = D/(6*ratio+1);
    rho = ratio * d;
    t = 0:pi/12:pi/2;
    pgon = polyshape({[-L/2, -L/2, -3*rho, -2*rho-rho*cos(t), L/2, L/2]}, {[0, D/2, D/2, D/2-2*rho-rho*sin(t), d/2, 0]});
end


tr = triangulation(pgon);
model = createpde('structural','static-planestress');

tnodes = tr.Points';
telements = tr.ConnectivityList';

geometryFromMesh(model,tnodes,telements);

generateMesh(model, 'Hmax', Hmax, 'Hmin', Hmin, 'GeometricOrder', 'linear');

mesh = model.Mesh;
structuralProperties(model,'YoungsModulus',70e9,'PoissonsRatio',0.25);
figure()
pdegplot(model, 'EdgeLabel', 'on');

structuralBC(model, 'Edge', [1, 5], 'YDisplacement', 0);
structuralBC(model, 'Edge', 4, 'XDisplacement', 0);
structuralBoundaryLoad(model,'Edge',2,'SurfaceTraction',[T;0]);
nodeConnectivity = mesh.Elements';
nodeCoordinate = mesh.Nodes';

end