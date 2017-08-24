function [ responses ] = SimulationEngine( params,designVars )
%SIMULATIONENGINE 
% params = [l,X,Y,E]    % length, load_x, load_y, Young's modulus
% designVars = [w,t]    % width, thickness
% responses = [A,R,D]   % Area, Stress, Displacement

% Unpack params and designVars
l = params(1);
X = params(2);
Y = params(3);
E = params(4);

w = designVars(1);
t = designVars(2);

A = w*t;
R = 600*Y/(w*t^2) + 600*X/(w^2*t);
D = 4*l^3 / (E*w*t) * sqrt((Y/t^2)^2 + (X/w^2)^2);

responses = [A,R,D];
end

