function [negloglike, nlls, pl, V_hist, omegaVs, RPE, Rel_diff] = funSideBias_DynamicOmega_decay(xpar,dat, initOmegaV)
% % funDQ_RPE % 
%PURPOSE:   Function for maximum likelihood estimation, called by fit_fun().
% AUTHOR: Jae Hyung Woo, 09/08/2023
%INPUT ARGUMENTS
%   xpar:       free parameters
%   dat:        data
%               dat(:,1) = chosen stimulus vector
%               dat(:,2) = reward vector
%               dat(:,3) = chosen location vector
%OUTPUT ARGUMENTS
%   negloglike:      the negative log-likelihood to be minimized
%   nlls      :      negative log-likelihood by trial
%   pl        :      model prob(choose left) by trial
%   V_hist    :      Value estimates histroy for all options 
%   omegaVs   :      omegaV values by trial
%   Rel (RPE) :      Reliablity of each system, i.e., RPE for each system 
%   Rel_diff  :      Reliablity difference between systems after sigmoid transform used to update omegaV

%% 
alpha       = xpar(1);    % learning rate for rewaraded options  
beta_DV     = xpar(2);    % inv. temp
SideBias    = xpar(3);    % constant side bias (preference for right)
alpha2      = xpar(4);    % learning rate for unrewaraded options  
decay_rate  = xpar(5);    % decay or forgetting rate for unchosen option  
gammaUpdate = xpar(6);    % fit update rate for omegaV
omega0      = xpar(7);    % initial omega at trial 1
decay_omega = xpar(8);    % decay rate for omega toward omega0

if exist('initOmegaV','var')
    omega0 = initOmegaV;    % if using input from previous block
end


omegaV      = omega0;

nt = size(dat,1);
negloglike = 0;

V = struct;     % initialie value func
V.Right = 0.5;  % value function for location
V.Left = 0.5;
V.Cir = 0.5;    % value function for shape
V.Sqr = 0.5;

V_hist.Stim1 = nan(nt,1);   % V_cir
V_hist.Stim2 = nan(nt,1);   % V_sqr
V_hist.Loc1 = nan(nt,1);    % V_left
V_hist.Loc2 = nan(nt,1);    % V_right
V_hist.DV_left = nan(nt,1); % total DV
V_hist.DV_right = nan(nt,1); % total DV

RPE.Stim = nan(nt,1);
RPE.Loc = nan(nt,1);
RPE.Combined = nan(nt,1);
Rel_diff = nan(nt,1);
omegaVs = nan(nt,1);    

choice_shape = dat(:,1);
reward = dat(:,2);
choice_location = dat(:,3);
shape_on_right = choice_shape.*choice_location;

pl = zeros(1,nt);
nlls = zeros(1,nt);

for k = 1:nt
%% looping through trials
    % assign omegaV weights
    omegaVs(k) = omegaV;    % store omegaV values for every trial
    omega_vS = omegaV;
    omega_vL = 1 - omegaV;
    
    % track record of all V's
    V_hist.Stim1(k) = V.Cir;
    V_hist.Stim2(k) = V.Sqr;
    V_hist.Loc1(k)= V.Left;
    V_hist.Loc2(k) = V.Right;
    
    % assign side
    switch shape_on_right(k)
        case -1
            V_Stim.right = V.Cir;
            V_Stim.left = V.Sqr;
        case 1
            V_Stim.right = V.Sqr;
            V_Stim.left = V.Cir;
    end
    DV_left = V_Stim.left*omega_vS + V.Left*omega_vL;
    DV_right = V_Stim.right*omega_vS + V.Right*omega_vL;
    
    % obtain final choice probabilities for Left and Right side
    [pleft, pright] = DecisionRuleSideBias2(SideBias,DV_left,DV_right,beta_DV);
    pl(k) = pleft;
    V_hist.DV_left(k) = DV_left;
    V_hist.DV_right(k) = DV_right;
    
    %compare with actual choice to calculate log-likelihood
    [nlls(k), negloglike] = NegLogLike(pleft,pright,choice_location(k),negloglike);
    
    % update value for the performed action:
    % Stimuli value functions
    [V.Cir, V.Sqr, rpe_stim] = IncomeUpdateStepRates(reward(k),choice_shape(k),V.Cir,V.Sqr,alpha,alpha2,decay_rate);
    % Location value functions
    [V.Left, V.Right, rpe_loc] = IncomeUpdateStepRates(reward(k),choice_location(k),V.Left,V.Right,alpha,alpha2,decay_rate);
    
    % update omegaV value based on reliability
    % Version : Use V_chosen as reliability signal
    deltaRel = rpe_loc - rpe_stim;
            % = V_chosen.Stim - V_chosen.Loc   : ranges [-1, 1], positive if Stim more reliable
    if deltaRel>0
       omegaV = omegaV + gammaUpdate*deltaRel*(1-omegaV) + decay_omega*(omega0 - omegaV); 
    else
       omegaV = omegaV + gammaUpdate*abs(deltaRel)*(0-omegaV) + decay_omega*(omega0 - omegaV); 
    end 
    % cap omegaV between [0 1]
    omegaV = max(0, omegaV);
    omegaV = min(1, omegaV);
    
    Rel_diff(k) = deltaRel;
    RPE.Stim(k) = rpe_stim;
    RPE.Loc(k) = rpe_loc;
    
    if choice_location(k)==-1
%         V_hist.DV_chosen(k) = DV_left;
        RPE.Combined(k) = reward(k) - DV_left;
    else
%         V_hist.DV_chosen(k) = DV_right;
        RPE.Combined(k) = reward(k) - DV_right;
    end
    
end

end