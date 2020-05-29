function [Qopt,meanError] = ksvd02(Y,T,Qopt0,plotsOn)
% Dictionary learning using K-SVD algorithm

[sigLength,nProfs]=size(Y);

nQopt = size(Qopt0,2);

% initializing dictionary using prescribed dictionary
Ynorm = Y./repmat(sqrt(diag(Y'*Y))',[sigLength,1]); % normaled examples

Qopt = Qopt0;

% solving for initial coefficients
xm = [];
xm = OMP_N(Qopt,Y,T);

% calculating initial error
error = [];
error(1,:)=mean(abs(Y-Qopt*xm));


%% dictionary learning with K-SVD algorithm
convCount = 0;
k = 1;
deltaThresh = .001;

while (convCount <= 5 && k <= 20)
    
    k = k+1;
    display(['iter # ',num2str(k-1)])
    
    % initialization of learning-loop variables
    qk = 1:nQopt;
    XU = xm;
    DU = Qopt;
    
    for n = 1:nQopt
        wk = find(XU(n,:));
        
        % replacing unpopular dictionary elements
        if length(wk) < 1
            dError = DU*XU-Y;
            dError = sum(dError.^2);
            [e,i] = max(dError);
            DU(:,n)=Ynorm(:,i);
            
        else
            % START YOUR CODE HERE <------------------------
            % Solve for DU and XU using K-SVD algorithm
            % move code to Matlab on your desktop to be able to see the original images and images of the dictionary
            % you obtain
            kInd = find(qk-n);  % the indices other than k 
            %dj =                % dicitonary atoms of these other indices
            %xj =                % corresponding coefficients for dj
            
            % You may use this space to help calculating Ek
            Qk = DU;
            Xk = XU;
            Xk(n,:) = [];
            Qk(:,n) = [];
            
            Ek = Y - Qk*Xk;               % Error matrix
            EkR = Ek(:,wk);     % Reduced error matrix
            [A,B,C] = svd(EkR);                    % You may use this line to implement SVD
            
            dk = A(:,1);             % updated dictionary atom
            DU(:,n) = dk;
            xk =C(:,1) * B(1:1);             % updated coefficient
            XU(n,wk) = xk;
            
        end
    end
    
    % sparse coding
    Qopt = DU;
    
    [xm,eOut] = OMP_N(Qopt,Y,T);
    error(k,:) = eOut;
    
    % determining convergence
    se = smooth(mean(error,2));
    me0 = se(k-1);
    me = se(k);
    delta = (me0-me)/me0;
    
    if abs(delta) < deltaThresh 
        convCount = convCount+1;
    else
        convCount = 0;
    end
    
    used = sum(XU~=0,2);
end

nIter = k-1;
%% plotting error vs. iterations
eMean = mean(error,2);
if plotsOn == true;
    figure(100); clf;
    plot([0:nIter],eMean)
    ylabel('Mean Error (m/s)')
    xlabel('Iteration #')
    grid on
end

meanError = eMean(end);

end