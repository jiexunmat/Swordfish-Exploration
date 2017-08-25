% Define ranges for our design variables
lb = [1,1];
ub = [4,4];

% Sampling with lhs
n_train = 500;
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

metamodel_method = 'GP';
% Time to build metamodel for each response
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

% Evaluate errors
errors = zeros(1, n_responses);
for i = 1:n_responses
    errors(i) = compute_RMSE(y_pred(:,i),y_test(:,i));
end