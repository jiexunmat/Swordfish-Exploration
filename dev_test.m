% Define ranges for our design variables
lb = [1,1];     % [width,thickness]
ub = [4,4];     % [width,thickness]

% Sampling with lhs
n_train = 400;
n_test = 1000;
x_train = lhs(lb,ub,n_train);
x_test = lhs(lb,ub,n_test);

% Collect responses
n_responses = 3;
y_train = zeros(n_train, n_responses);
for i = 1:n_train
    y_train(i,:) = SimulationEngine(x_train(i,:));
end
y_test = zeros(n_test, n_responses);
for i = 1:n_test
    y_test(i,:) = SimulationEngine(x_test(i,:));
end

% Build metamodel for each response
gpoptions.covfunc = {'covSum', {'covSEard','covNoise'}};
for i = 1:n_responses
    gpdata(i) = gaussianprocessregression('Train',x_train,y_train(:,i),gpoptions);
end


% Make predictions on x_test
y_pred = zeros(n_test, n_responses);
for i = 1:n_responses
    y_pred(:,i) = gaussianprocessregression('Evaluate', x_test, gpdata(i)); 
end

% Evaluate errors
errors = zeros(1, n_responses);
for i = 1:n_responses
    errors(i) = compute_RMSE(y_pred(:,i),y_test(:,i));
end

% Surf plot for objectives and constraints
[X,Y] = meshgrid(1:0.05:4, 1:0.05:4);
A = reshape(X,[numel(X),1]);
B = reshape(Y,[numel(Y),1]);
C = [A,B];
y_grid = zeros(numel(X), n_responses);
for i = 1:n_responses
    y_grid(:,i) = gaussianprocessregression('Evaluate', C, gpdata(i)); 
end

Wt = reshape(y_grid(:,1), size(X));
S = reshape(y_grid(:,2), size(X));
D = reshape(y_grid(:,3), size(X));

S_lim = ones(size(S)) * 4E4;
D_lim = ones(size(D)) * 2.2535;

S_bool = S<=S_lim;
S_clip = S.*S_bool;
D_bool = D<=D_lim;
D_clip = D.*D_bool;

figure
set(gcf, 'Units', 'normalized', 'Position', [0.05, 0.2, 0.9, 0.4])
subplot(1,3,1)
surf(X,Y,Wt)
title('Weight')
xlabel('width,w'); ylabel('thickness,t');

subplot(1,3,2)
surf(X,Y,S_clip); hold on;
%surf(X,Y,S_lim)
title('Stress')
xlabel('width,w'); ylabel('thickness,t');

subplot(1,3,3)
surf(X,Y,D_clip); hold on;
%surf(X,Y,D_lim)
title('Displacement')
xlabel('width,w'); ylabel('thickness,t');


