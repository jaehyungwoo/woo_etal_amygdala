function [negloglike, nlls, pl, V_hist, RPE] = funSideBias_StaticOmega(xpar,dat)
% % funDQ_RPE % 
% AUTHOR: Jae Hyung Woo, 06/19/2023
%PURPOSE:   Function for maximum likelihood estimation, called by fit_fun().
%
%INPUT ARGUMENTS
%   xpar:       free parameters (see below script for description)
%   dat:        data
%               dat(:,1) = chosen stimulus vector
%               dat(:,2) = reward vector
%               dat(:,3) = chosen location vector
%OUTPUT ARGUMENTS
%   negloglike:      the negative log-likelihood to be minimized
%   V_delta   :      struct for storing value estimate difference between two option, d = V_1 - V_2
%                    for Stim & Loc dimensions
%                    Stim_c & Loc_c = V_chosen - V_unchosen   
%   RPE       : RPE for each system

%% 
alpha       = xpar(1);      % learning rate for rewaraded options
betaDV      = xpar(2);      % inverse temperature
SideBias    = xpar(3);      % constant side bias (preference for right)
alpha2      = xpar(4);      % learning rate for unrewaraded options
decay_rate  = xpar(5);      % decay or forgetting rate for unchosen option
omega       = xpar(6);      % fixed omega weight

omega_vS    = omega;
omega_vL    = 1 - omega;

nt = size(dat,1);
negloglike = 0;

vL_right = 0.5;  % value function for location
vL_left = 0.5;
vS_cir = 0.5;    % value function for shape
vS_sqr = 0.5;

V_hist.Stim1 = nan(nt,1);   % V_cir
V_hist.Stim2 = nan(nt,1);   % V_sqr
V_hist.Loc1 = nan(nt,1);    % V_left
V_hist.Loc2 = nan(nt,1);    % V_right
V_hist.DV_left = nan(nt,1); % total DV
V_hist.DV_right = nan(nt,1); % total DV
RPE.Stim = nan(nt,1);
RPE.Loc = nan(nt,1);

choice_shape = dat(:,1);
choice_location = dat(:,3);
shape_on_right = choice_shape.*choice_location;

pl = zeros(1,nt);
nlls = zeros(1,nt);

for k = 1:nt
%% Loop through trials
    % track record of all V's
    V_hist.Stim1(k) = vS_cir;
    V_hist.Stim2(k) = vS_sqr;
    V_hist.Loc1(k)= vL_left;
    V_hist.Loc2(k) = vL_right;
    
    % assign side
    switch shape_on_right(k)
        case -1
            vS_right = vS_cir;
            vS_left = vS_sqr;
        case 1
            vS_right = vS_sqr;
            vS_left = vS_cir;
    end
    q_left = vS_left*omega_vS + vL_left*omega_vL;
    q_right = vS_right*omega_vS + vL_right*omega_vL;
    
    % obtain final choice probabilities for Left and Right side
    [pleft, pright] = DecisionRuleSideBias2(SideBias,q_left,q_right,betaDV);
    pl(k) = pleft;
    V_hist.DV_left(k) = q_left;
    V_hist.DV_right(k) = q_right;
    
    %compare with actual choice to calculate log-likelihood
    [nlls(k), negloglike] = NegLogLike(pleft,pright,choice_location(k),negloglike);
    
    % update value for the performed action:
    % Stimuli value functions
    [vS_cir, vS_sqr, RPE.Stim(k)] = IncomeUpdateStepRates(dat(k,2),choice_shape(k),vS_cir,vS_sqr,alpha,alpha2,decay_rate);
    % Location value functions
    [vL_left, vL_right, RPE.Loc(k)] = IncomeUpdateStepRates(dat(k,2),choice_location(k),vL_left,vL_right,alpha,alpha2,decay_rate);
end

end