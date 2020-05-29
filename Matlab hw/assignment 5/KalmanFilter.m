function [ x_hat, P_hat, K ] = KalmanPG( y, sys)

%The simplest Kalman filter
%   x_p   = M x_hat+ N(0,Q)
%   x_hat = H x_p +N(0,R)

% dimensions
% dimensions
[Nobs, Ntime] = size(y);
Nx = length(sys.x0);
M  = sys.M; % state eq matrix
H  = sys.H; % observation eq matrix
R  = sys.R; % observation noise
Q  = sys.Q; % prediction noise
%%
K     = zeros(Nx,Nobs,Ntime);  
P_p   = zeros(Nx,Nx,Ntime);
P_hat = zeros(Nx,Nx,Ntime);
x_hat = zeros(Nx,Ntime);
x_p   = zeros(Nx,Ntime);
%initial states
x_hat(:,1)    = sys.x0;
P_hat(:,:,1)  = sys.P0;
for iobs=2:Ntime
    % predict
    
    x_p(:,iobs)         = M * x_hat(:, iobs-1);% Predicted State Estimate
    P_p(:,:,iobs)       = M * P_p(:,:,iobs-1) * (transpose(M)) + Q;% Predicted Error Covariance
    % update
    Ptemp               = P_p(:,:,iobs);
    K(:,:,iobs)         = Ptemp * (transpose(H)) * inv((H*Ptemp*(transpose(H))+R)); % Kalman Gain
    P_hat(:,:,iobs)     = (1-K(:,:,iobs)*H)*P_hat(:,:,iobs);% Updated Estimate Covaraince
    x_hat(:,iobs)       = x_hat(:,iobs)+K(:,:,iobs)*(y(:,iobs) - H*x_hat(:,iobs));% Updated State Estimate 
end

end