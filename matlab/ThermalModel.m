function [model] = ThermalModel(k, rho, cp)
    % Geometry
    L = 0.5;
    r = 0.25;
    t = pi/24:pi/24:2*pi;
    pgon = polyshape({[-L, -L, L, L], [r*cos(t)] }, {[-L, L, L, -L], [r*sin(t)]});

    % Mesh
    tr = triangulation(pgon);
    tnodes = tr.Points';
    telements = tr.ConnectivityList';
    model = createpde('thermal','transient');
    geometryFromMesh(model,tnodes,telements);
    Hmax = 0.03;
    Hmin = 0.01;
    generateMesh(model, 'Hmax', Hmax, 'Hmin', Hmin, 'GeometricOrder', 'linear');
    
    thermalProperties(model,'ThermalConductivity',k, 'MassDensity',rho, 'SpecificHeat',cp);
end

