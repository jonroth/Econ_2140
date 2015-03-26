%This function regresses y on X. It includes a constant in X if const=1. It
%returns the estimated coefficients beta, the estimated standard errors se,
%as well as the variance-covariance matrix for beta-hat, Sigma (this is for
%beta-hat, not sqrt(n)*beta-hat).
%This function was written for Ec 2140, and used in Ec 2120 as well.
function[beta, se, Sigma, se_homo, Sigma_homo] = reg(y,X, const)

%If constant is 1, add a column of ones to the X matrix
    if const == 1
        cvec = ones(length(X),1);
        X = [cvec X];
    end
    
    %Calculate beta using standard ols formula
    beta = inv(X' * X) * X' * y;

    %Calculate the residuals and square residuals
    u = y - X*beta;
    u2 = u.*u;

    %Calculate robust standard errors
    
    %First create a matrix with the squared residuals on the diagnol
    U = diag(u2);
    %Now, calculate the sample variance covariance matrix (note this is the
    %variance covariance for Beta, not for N^(1/2) * beta)
    Sigma = inv(X'* X) * (X'*U*X) * inv(X'*X);
    %Calculate the SEs as sqrt of diagnol
    se = sqrt(diag(Sigma));
    
    n = length(y);
    k = size(X,2);
    Sigma_homo = 1/(n-k)*sum(u2)* inv(X'* X);
    se_homo = sqrt(diag(Sigma_homo));
end
