function varargout = polynomialregression(Action,order,X,varargin)
%POLYNOMIALREGRESSION Low level polynomial regression function.
%
%       COEFFS = POLYNOMIALREGRESSION('Train',ORDER,X,Y)
%       Returns the coefficients that make || Y - X*coeffs || minimal.
%       In X and Y each row represents a data point and each column a
%       dimension of the inputs or outputs respectively.
%
%       YHAT = POLYNOMIALREGRESSION('Evaluate',ORDER,X,COEFFS)
%       Returns the polynomial evaluation at X. In X each row represents a
%       data point and each column a dimension of the input. COEFFS should
%       have the format as when returned by this function.
%
%       Author(s): R. Frigola, 28-04-09
%       Copyright (c) 2009 McLaren Racing Limited.
%       $Revision: 1 $  $Date: 28/04/09 15:36 $ $Author: roger.frigola $  


onesvector = ones(size(X,1),1);

% Build alternant matrix
switch order
    case 1
        % [Linear, Constant]
        alternantMatrix = [X  onesvector];
    case 2
        % [Quadratic, Bilinear, Linear, Constant]
        alternantMatrix = [X.*X  bilinearterms(X)  X  onesvector];
    otherwise
        % CHALLENGE: find a neat, generic, function that can be used to
        % compute the alternant matrix for an arbitrary order.
        merror('Store', 0, ['Swordfish: Polynomial order ' num2str(order) ' not yet supported.']);
        merror('Raise');
end

switch Action
    case 'Train'
        % Return NaN coefficients if input is empty
        if isempty(X)
            coeffs = NaN * ones(size(alternantMatrix,2),1);
            varargout{1} = coeffs;
            return
        end
        % Check enough data points to train the metamodel
        % i.e. dimMetamodel > numDataPoints
        if size(alternantMatrix,2) > length(onesvector)
            errordlg(['There are not enough data points to fit a metamodel for one of the targets. This metamodel has ' ...
                    num2str(size(alternantMatrix,2)) ' coefficients and there are only ' num2str(length(onesvector)) ...
                    ' experiments (numTotalExpermients/numEnumerativeVarCombinations).'],'Not enough data points to fit metamodel');
        end
        
        Y = varargin{1};
        coeffs = alternantMatrix \ Y;
        varargout{1} = coeffs;
    case 'Evaluate'
        coeffs = varargin{1};
        Yhat = alternantMatrix * coeffs;
        varargout{1} = Yhat;
end


        
% -------------------------------------------------------------------------
function X = bilinearterms(A)
% Generates all the possible bilinear terms of the columns of a matrix. 
% Note that it does not generate the pure quadratic terms.
%
% Exmaple: A=[2 7 3] then X=[2*7 2*3 7*3]
%
% RF, 2008

persistent n iia iib

if isempty(n)
    n = size(A,2);
end
m = size(A,1);

% Find indices of bilinear terms if they haven't been computed yet or they
% have been computed for a matrix of different size
if isempty(iia) | n~=size(A,2)
    n = size(A,2);
    iia = zeros((n^2-n)/2 , 1 );
    iib = zeros((n^2-n)/2 , 1 );
    count = 1;
    for i=1:n
        for j=i+1:n
            iia(count) = i;
            iib(count) = j;
            count = count + 1;
        end
    end     
end
X = zeros(m, (n^2-n)/2 ); % Marginally faster preallocating memory
X = A(:,iia) .* A(:,iib);
