function fhat = f_kern(grid, X, Kern, h)
    %save the length of X as n, length of grid as J
    n = size(X,1);
    J = size(grid,1);
    
    %Create a matrix devmat where the ij'th entry is 1/h*(X_j - g_i)
    devmat = 1/h*(repmat(X',J,1) - repmat(grid,1,n) );
    
    %Evaluate K at the deviations (K should take a matrix and return
    %elementwise results)
    Kweights = Kern(devmat);
    
    %Sum over the columns to get the weight for each point in the grid
    fhat = 1/(n*h)*sum(Kweights, 2);

    

end