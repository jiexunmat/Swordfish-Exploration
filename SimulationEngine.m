function [ responses ] = SimulationEngine( designVars )
%SIMULATIONENGINE 
% params = [l,X,Y,E]    % length, load_x, load_y, Young's modulus
% designVars = [w,t]    % width, thickness
% responses = [S,R,D]   % Area, Stress, Displacement

% Simulation parameters
l = 100;
X = 500;
Y = 1000;
E = 2.9E7;

% Unpack designVars
w = designVars(1);
t = designVars(2);

% Calculate responses
A = w*t;
S = 600*Y/(w*t^2) + 600*X/(w^2*t);
D = 4*l^3 / (E*w*t) * sqrt((Y/t^2)^2 + (X/w^2)^2);

responses = [A,S,D];
end

