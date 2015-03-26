%This function takes a panel of observations Q (where each row is a
%household), and a matrix of predictors R (again, where each row is a HH),
%and returns the generalized linear predictor using weight matrix Phi
function beta = genlinpred(Q, R, Phi)

beta = (R'*Phi*R)\(R'*Phi*Q);


end