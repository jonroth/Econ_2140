function [g H] = duration_gh(alpha, beta, Y, X, Z)

n = length(Y);
k = length(beta);

%initialize g and H as zero, then add to these in the sum

g = zeros(k+1,1);
H = zeros(k+1,k+1);

%i = 1;
for i = 1:n
    
    %Take y,x,z for the specific i
    y_i = Y(i);
    x_i = X(i,:)';
    z_i = Z(i);
    
    %compute the components of the gradient
    dl_da = (1 - z_i) * (1/alpha + log(y_i)) - exp(x_i' *beta) * log(y_i) * y_i^alpha;
    dl_db = ( (1 - z_i) - exp(x_i' *beta) * y_i^alpha ) * x_i;
    
    %Add to the gradient sum
    g = g + [dl_da ; dl_db];
    
    %Compute the components of the Hessian
    d2l_dbdb = - exp(x_i'*beta)*y_i^alpha * (x_i * x_i');
    d2l_dada = - (1 - z_i)/alpha^2 - exp(x_i'*beta) * log(y_i)^2 * y_i^alpha;
    d2l_dbda = - exp(x_i' * beta) * log(y_i) * y_i^alpha * x_i;
    
    H = H + [d2l_dada, d2l_dbda'; d2l_dbda, d2l_dbdb];
end


end