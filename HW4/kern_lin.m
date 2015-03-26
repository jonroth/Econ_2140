%This function takes a grid of points to evaluate mhat(g), the covariates X
%and outcomes Y, a Kernel function Kern, and a bandwidth. It returns the
%predictions mhat(g) for all g in grid using local linear regression.
%It also returns Kweights, a matrix where the ij'th entry is the weight on
%Y_j for g_i.


function  [mhat, Weights] = kern_lin(grid, X, Y, Kern, h)
%function  w = kern_lin(grid, X, Y, Kern, h)


    %mhat = w_i(grid(1), X, Y, Kern, h);
    %Weights = 0;
    
    %Apply the w_i function below to all the elements of g. These are the
    %rows of the weighting matrix. Then stack them into the W matrix
    
    wcell = arrayfun( @(g) w_i(g, X, Y, Kern, h), grid, 'UniformOutput', false);
    Weights = cell2mat(wcell);

    %Compute the predicted values of Y
    mhat = Weights * Y;

    
end


%Create a helper function that computes the weight vector for a scalar
%value of g; i.e., one row of the weight matrix

function w_i = w_i(g, X, Y, Kern, h)
    
    %Store length of X as 1
    n = size(X,1);
    
    %Get the vector of Kweights using the kern_nw function
    [~,Kvec] = kern_nw(g, X, Y, Kern, h);
    
    z = [1; g];
    Z = [ones(n,1), X];
    
    A =  z'* (Z' * diag(Kvec) * Z )^(-1);
    
    A_rep = repmat(A,n,1);
    
    B_rep = repmat(Kvec',1,length(z));
    
    C_rep = [ones(1,n); X'];
    
    w_i = diag(A_rep .* B_rep * C_rep)';
    
end

