%This function uses a Newton Raphson method to find theta, given a score
%function g, a Hessian function H, an initial guess for theta, and a
%tolerance tol. (function g should be a column vec)
function [theta] = newtonraphson(g,H, guess, tol)

%Initialize theta and the error term
theta = guess;
err = tol + 1;

%While the error exceeds tol, update theta, compute error
while err > tol
   
   %calculate new value of theta
   thetanew = theta - inv(H(theta)) * g(theta); 
   %compute err as the change in theta from last iteration
   err = max(abs(thetanew - theta));
   %update the theta variable
   theta = thetanew;
    
end

end