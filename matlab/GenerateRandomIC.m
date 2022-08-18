function [T0] =GenerateRandomIC(model)
    mesh = model.Mesh;
    nodeCoordinate = mesh.Nodes';
    corr.name = 'exp';
    corr.c0 = [0.1, 0.1]; % anisotropic correlation
    corr.sigma = 1;
    mesh = nodeCoordinate;
    [F,~] = randomfield(corr,mesh);

    F_interp = scatteredInterpolant(nodeCoordinate(:, 1), nodeCoordinate(:, 2), F, 'natural');
    T0 = @(location) F_interp(location.x, location.y);
end

