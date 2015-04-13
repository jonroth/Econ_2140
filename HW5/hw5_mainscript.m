cd '/Users/jonathanroth/Google Drive/Econ 2140/HW5/';

%Load the file
Y = load('psid.dat.txt');

%Save n and T
T = size(Y,2);
n = size(Y,1);

%% Pooled regression

%Create Ydif as the matrix of first differences (t = T,...3), and Ydif_l1
%as the first differences (t = T-1,...,2)
Ydif_mat = diff(Y,1,2);
Ydif = Ydif_mat(:,2:(T-1));
Ydif_l1 = Ydif_mat(:,1:(T-2));

%Run the pooled regression
[rho,rho_sd] = reg(Ydif(:),Ydif_l1(:),0);

%% GMM
%Construct the Z matrix
Z_cell = cell(n,1);

for i = 1:n
    
    Z_i_cell = cell(1,T-2);
    
   for t = 3:T
       
       y_t = Y(i,1:(t-2));
       Z_i_cell{1,t-2} = y_t;
   end
   
   Z_cell{i,1} = blkdiag(Z_i_cell{:});
end

Z = cell2mat(Z_cell);

%Construct the X matrix by stacking the rows of Ydif_l1
X = Ydif_l1';
X = X(:);

%Construct the matrix DY by stacking the rows of Ydif
DY = Ydif';
DY = DY(:);

%Construct G
%G = full(spdiags(-ones(T-2,2), [-1,1], 2*eye(T-2)));
G = spdiags(-ones(T-2,2), [-1,1], 2*eye(T-2));

%Calculate the homoskedastic weight matrix
W_1 = (Z' * kron(eye(n),G) * Z)^(-1);

%Create a function that comptues rho hat given the data and a weighting mat
rho_hat_fn = @(X,Z,W,DY) (X'*Z* W * Z' *X)^(-1) * X'*Z*W*Z'* DY;

%Evaluate rho1 using W_1
rho1 = rho_hat_fn(X,Z,W_1,DY)


%compute the residuals
uhat = DY - X*rho1;

%Calculate the homoskedastic variance
s2 = 1/(n*(T-2)) * sum(uhat.^2);
varmat_homo = s2*(X'*Z* W_1 * Z' *X)^(-1);
se_homo = sqrt(varmat_homo)


%Construct hte optimal matrix using the residuals from the first estimator
W_2 = (Z' * diag(uhat.^2) * Z)^(-1);


rho_opt = rho_hat_fn(X,Z,W_2,DY)


%Calculate heteroskedastic variance

%compute the residuals for optimal matrix
uhat_opt = DY - X*rho_opt;


rho_hat_het = @(X,Z,W,u) (X'*Z* W * Z' *X)^(-1) * (Z'*X)' * W* (Z' * diag(u.^2) * Z) * W * (Z'*X)...
    * (X'*Z* W * Z' *X)^(-1);

rho_hat_het2 = @(X,Z,W,u) (X'*Z* W * Z' *X)^(-1);

var_het = rho_hat_het(X,Z,W_2,uhat_opt)

var_het2 = rho_hat_het2(X,Z,W_2,uhat_opt)

se_het = sqrt(var_het)
se_het2 = sqrt(var_het2)

%% Two lags

%Create Ydif as the matrix of first differences (t = T,...4), and Ydif_l1
%as the first differences (t = T-1,...,3), and Ydif_l2 (t = T,...,2)
Ydif = Ydif_mat(:,3:(T-1));
Ydif_l1 = Ydif_mat(:,2:(T-2));
Ydif_l2 = Ydif_mat(:,1:(T-3));



%Construct the Z matrix
Z_cell = cell(n,1);

for i = 1:n
    
    Z_i_cell = cell(1,T-3);
    
   for t = 4:T
       
       y_t = Y(i,1:(t-2));
       Z_i_cell{1,t-2} = y_t;
   end
   
   Z_cell{i,1} = blkdiag(Z_i_cell{:});
end

Z = cell2mat(Z_cell);


%Construct the X matrix by stacking the rows of Ydif_l1, Ydif_l2
X1 = Ydif_l1';
X2 = Ydif_l2';
X = [X1(:) , X2(:)];


%Construct the matrix DY by stacking the rows of Ydif
DY = Ydif';
DY = DY(:);

%Construct G
%G = full(spdiags(-ones(T-2,2), [-1,1], 2*eye(T-2)));
G = spdiags(-ones(T-3,2), [-1,1], 2*eye(T-3));

%Calculate the homoskedastic weight matrix
W_1 = (Z' * kron(eye(n),G) * Z)^(-1);

%calculate the coefficient
rho_vec1 = rho_hat_fn(X,Z,W_1,DY)

%Calculate the homoskedastic variance
%varmat_homo2 = (X'*Z* W_1 * Z' *X)^(-1);
%se_homo2 = sqrt(diag(varmat_homo2))

%compute the residuals
uhat1 = DY - X*rho_vec1;

%Calculate the homoskedastic variance
s2 = 1/(n*(T-3)) * sum(uhat1.^2);
varmat_homo2 = s2*(X'*Z* W_1 * Z' *X)^(-1);
se_homo2 = sqrt(diag(varmat_homo2))
