
%This function takes a vector y, an endogenous variable x, and an
%instrument z, and runs 2SLS. The function includes a constant, with beta0
%the first element in the returned vector
function betahat = twosls(y,x,z)

%Set n to the number of obs
n = length(y);

%Set zvec to z and a constant
zvec = [ones(n,1) z];
%regress x on z and a constant, form predicted xhat
gamma = (zvec'*zvec)^(-1)*(zvec'*x);
%form the predicted values of logp,and put them in a vector with a constant
%call them xhat
xhat = [ones(n,1) zvec*gamma];

%Compute the 2sls estimator
betahat = (xhat'*xhat)^(-1)*(xhat'*y);

%compute the alternate 2sls estimator
betahat2 = (xhat'*[ones(n,1) x])^(-1) * (xhat'*y);

end