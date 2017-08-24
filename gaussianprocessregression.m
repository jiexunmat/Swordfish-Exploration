function varargout = gaussianprocessregression(Action,X,varargin)
%GAUSSIANPROCESSREGRESSION Low level gaussian process regression function.
%
%       GPDATA = GAUSSIANPROCESSREGRESSION('Train',X,Y,GPOPTIONS)
%       In X each row represents a data point and each column a
%       dimension of the inputs. The monodimensional output Y needs to be
%       arranged as a column vector. The argument GPOPTIONS contains 
%       information on how the gaussian process must be trained.
%       GPOPTIONS needs to have the following field:
%           - covfunc: Covariance functions, e.g. {'covSum', {'covSEard','covNoise'}}
%       And can optionally have the fields:
%           - logtheta0: Initial guess of the hyperparameters
%           - algorithm: Specifies the algorithm used to find the
%             hyperparameters. If 'Knitro' is selected, uses mknitro with
%             analytical derivatives.
%           - numMaxFunEvaluations: Specifies the maximum number of
%             function evaluations if using the default algorithm
%             'minimise'
%       GPDATA is a structure containing the original training set,
%       information about the covariance function, the values of the
%       hyper-parameters and other parameters needed to evaluate the
%       gaussian process.
%
%       YHAT = GAUSSIANPROCESSREGRESSION('Evaluate',X,GPDATA)
%       Returns the evaluation at X. In X each row represents a
%       data point and each column a dimension of the input.
%
%       [YHAT,S2] = GAUSSIANPROCESSREGRESSION('Evaluate',X,GPDATA)
%       Returns the variance as well. This is the gaussian process
%       assessment of its own variance and it should not be blindly
%       trusted.
%
%       Author(s): R. Frigola, 04-06-09
%       Copyright (c) 2009 McLaren Racing Limited.
%       $Revision: ? $  $Date: ? $ $Author: roger.frigola $  
%
%       Revised in 18/08/2017 to make use of gpml matlab code v4.0

% gpoptions.sparse denotes if sparse or full GP is to be used

switch Action
    case 'Train'
        y = varargin{1};
        gpoptions = varargin{2};
        
        % Check that there is more than one training point
        if numel(y) == 1
            progress('Close', progressId)
            errordlg('There is little to be gained by fitting a metamodel to a single point...','Only one point to fit metamodel');
            merror('Store', 0, ['Swordfish: There is only one point in the training set.']);
            merror('Raise');
        end
        
        % Demean
        offset = mean(y);
        y = y - offset;
        
        % Are we in Swordfish? 
        dbs = dbstack;
        bSwordfish = numel(dbs)>1 && strcmp(dbs(2).file,'metamodel.m');
        
        % Check if the covariance function covfunc is the Automatic
        % Relevance Determination function with noise.
        bARDNoise = numel(gpoptions.covfunc)==2 && numel(gpoptions.covfunc{2})==2 && ...
            strcmp(gpoptions.covfunc{2}{1},'covSEard') && strcmp(gpoptions.covfunc{2}{2},'covNoise');
        
        % Initial guess for the hyperparameter optimization, we make a
        % distinction depending on whether we are in Swordfish or not.
        if isfield(gpoptions,'logtheta0')
            logtheta0 = gpoptions.logtheta0(:);
        
        % DEV: always use this method, not ARD
        %elseif ~bSwordfish && bARDNoise
        elseif bARDNoise
            % Lengthscales are based on the standard deviation of the
            % inputs. We guess a signal to noise ratio of 20:1.
            stdX = std(X)';
            stdX( stdX./abs(mean(X))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
            logtheta0 = log([stdX; std(y); 0.05*std(y)]);
            
        % DEV: don't use ARD
%         elseif bSwordfish && bARDNoise
%             % Automatic Relevance Determination
%             % http://www.gaussianprocess.org/gpml/code/matlab/doc/regression.html
%             % logtheta = [log(ell_1), ..., log(ell_dimInputs),
%             % log(sigma_f), log(sigma_n)]
%             logtheta0 = [ones(dimInputs,1); log(std(y)); log(std(y)/10)];  % Minimisation starting point
        else
            error('The field logtheta0 is necessary in GPOPTIONS for this kind of covariance function')
        end
        
        %% Setting up options
        if isfield(gpoptions,'numMaxFunEvaluations')
            numMaxFunEvaluations = gpoptions.numMaxFunEvaluations;
        else
            numMaxFunEvaluations = 50;
        end
        if isfield(gpoptions,'sparse')
            using_sparse = gpoptions.sparse;
        else
            using_sparse = 0;
        end
        
        % We use a zero mean function, the default inference method, 
        % likelihood measure and covariance function.
        inf = @infGaussLik;
        empty_mean = [];
        cov = {@covSum, {@covSEard,@covNoise}};
        lik = {@likGauss};
        hyp.cov = logtheta0;
        hyp.lik = logtheta0(end);
        
        if using_sparse 
            if isfield(gpoptions,'numInducingPoints')
                numInducingPoints = gpoptions.numInducingPoints;
            else
                numInducingPoints = 200;
            end
            if isfield(gpoptions, 'alpha_s') % parameter to switch between VFE, SPEP, FITC
                alpha_s = gpoptions.alpha_s;
            else
                alpha_s = 0.5;
            end
            
            % Initialise inducing points randomly as a subset of training data
            n_train = size(X,1);
            indices = randperm(n_train);
            sparse_indices = indices(1:numInducingPoints);
            xu = X(sparse_indices,:);
            
            hyp.xu = xu;
            %inf = @(varargin) inf(varargin{:},struct('s',alpha_s));
            cov = {'apxSparse',cov,xu};
        end
       
        %% Learn hyperparameters by minimising the negative log marginal likelihood
        hyp = minimize_minfunc(hyp,@gp,-numMaxFunEvaluations,inf,empty_mean,cov,lik,X,y);
        [~, ~, post] = gp(hyp, inf, empty_mean, cov, lik, X, y);
        
        % Save hyperparameters into gpdata
        if using_sparse
            gpdata.xu = hyp.xu;
        end
        gpdata.x = X;
        gpdata.y = y;
        gpdata.hyp = hyp;
        gpdata.inf = inf;
        gpdata.mean = empty_mean;
        gpdata.cov = cov;
        gpdata.lik = lik;
        gpdata.L = post.L;
        gpdata.alpha = post.alpha;
        gpdata.offset = offset;
        gpdata.logtheta = hyp.cov;
        
        varargout{1} = gpdata;

    case 'Evaluate'
        gpdata = varargin{1};
        mu = gp(gpdata.hyp, gpdata.inf, gpdata.mean, gpdata.cov, gpdata.lik, gpdata.x, gpdata.y, X);
        varargout{1} = mu + gpdata.offset;
end

%--------------------------------------------------------------------------
%   PRIVATE FUNCTIONS
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
function S = lhs(xMin,xMax,nSample)
% LHS  Latin Hypercube Sampling
% Input:
%   xMin    : vector of minimum bounds
%   xMax    : vector of maximum bounds
%   nSample : number of samples
% Output:
%   S       : matrix containing the sample (nSample,numVar)
%
% RF, 20/04/2009

numVar = length(xMin);
ran = rand(nSample,numVar);
S = zeros(nSample,numVar);
for i=1:numVar
   idx = randperm(nSample);
   P = (idx'-ran(:,i))/nSample;
   S(:,i) = xMin(i) + P.* (xMax(i)-xMin(i));
end
S = S(randperm(size(S,1)),:);