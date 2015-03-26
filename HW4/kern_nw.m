%This function takes a grid of points to evaluate mhat(g), the covariates X
%and outcomes Y, a Kernel function Kern, and a bandwidth. It returns the
%predictions mhat(g) for all g in grid using the Nadaraya Watonson kernel
% regression. It also returns Kweights, a matrix
%where the ij'th entry is the weight on Y_j for g_i.

function [mhat, Kweights] = kern_nw(grid, X, Y, Kern, h)

    %save the length of X as n, length of grid as J
    n = size(X,1);
    J = size(grid,1);
    
    %Create a matrix devmat where the ij'th entry is 1/h*(X_j - g_i)
    devmat = 1/h*(repmat(X',J,1) - repmat(grid,1,n) );
    
    %Evaluate K at the deviations (K should take a matrix and return
    %elementwise results)
    Kweights = Kern(devmat);
    
    %Normalize the rows so that they sum to 1
    Kweights = Kweights ./ repmat( sum(Kweights,2) ,1,n);
    
    %Compute the predicted values of Y
    mhat = Kweights * Y;

end