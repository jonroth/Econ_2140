function CV = cross_val(X, Kern, K2, h)


    %save the length of X as n
    n = size(X,1);
    
    %Create a matrix devmat where the ij'th entry is 1/h*(X_j - g_i)
    devmat = 1/h*(repmat(X',n,1) - repmat(X,1,n) );
    
    K2_mat = K2(devmat);
    
    %The first part of CV is 1/(N^2*h) times the sum of all the deviations
    %evaluated with K2
    CV_1 = 1/(n^2*h) * sum(K2_mat(:));
    
    %Now 




end