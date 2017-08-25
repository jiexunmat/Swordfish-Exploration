% Define ranges for our design variables
lb = [1,1];     % [width,thickness]
ub = [4,4];     % [width,thickness]

% Basic parameters
SIMULATION_BUDGET = 500;
n_train = SIMULATION_BUDGET*0.7;
n_test = SIMULATION_BUDGET*0.3;

%% Design of Experiments (Sampling scheme)
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

%% Metamodelling
metamodel_method = 'GP';
% Build metamodel for each response
y_pred = zeros(n_test, n_responses);
switch metamodel_method
    case 'GP'
        gpoptions.covfunc = {'covSum', {'covSEard','covNoise'}};
        for i = 1:n_responses
            gpdata(i) = gaussianprocessregression('Train',x_train,y_train(:,i),gpoptions);
        end
        for i = 1:n_responses
            y_pred(:,i) = gaussianprocessregression('Evaluate', x_test, gpdata(i)); 
        end
    case 'Poly'
        for i = 1:n_responses
            all_coeffs(i).coeffs = polynomialregression('Train',2,x_train,y_train(:,i));
        end
        for i = 1:n_responses
            y_pred(:,i) = polynomialregression('Evaluate', 2, x_test, all_coeffs(i).coeffs); 
        end
end

% Obtain validation errors
errors = zeros(1, n_responses);
for i = 1:n_responses
    errors(i) = compute_RMSE(y_pred(:,i),y_test(:,i));
end
fprintf('\n--------------------------------\n');
fprintf('Number of simulation calls = %d\n', SIMULATION_BUDGET);
ave_resp = sum(y_test,1)./size(y_test,1);
disp('Percentage errors on each response:')
disp(errors./ave_resp*100)

%% Visualisation of response surface
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
view([0,90])
title('Weight')
xlabel('width,w'); ylabel('thickness,t');

subplot(1,3,2)
surf(X,Y,S_clip); hold on;
view([0,90])
%surf(X,Y,S_lim)
title('Stress')
xlabel('width,w'); ylabel('thickness,t');

subplot(1,3,3)
surf(X,Y,D_clip); hold on;
view([0,90])
%surf(X,Y,D_lim)
title('Displacement')
xlabel('width,w'); ylabel('thickness,t');
