%Take X, Y, and a function MW. MW is a function such as kern_nw. It takes
% as inputs (grid, X, Kern, h); and returns the predictions mhat(grid) as
% well as the weights matrix associated with these estimates

function cv = reg_cv(X, Y, MW, Kern, h)

    [mx, Kweights] = MW(X,X,Y,Kern,h);

    n = length(mx);
    
    %Initialize cv to 0
    cv = 0;
    
    for i = 1:n
        y_i = Y(i);
        mx_i = mx(i);
        w_i = Kweights(i,i);
        cv_i = ((y_i - mx_i)/(1 - w_i))^2;
        
        cv = cv + cv_i;
        
    end
    
    cv = cv/n;
end