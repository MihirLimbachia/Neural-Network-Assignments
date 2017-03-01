% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc

Ntrain=load('approximationproblems\SI.tra');
[NTD,~] = size(Ntrain);

% Initialize the Algorithm Parameters.....................................
inp = 2;            % No. of input neurons
hid = 10;                % No. of hidden neurons
out = 1;            % No. of Output Neurons
lam = 1.e-02;       % Learning rate
epo=20;

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        Wo = Wo + lam * (er * Yh'); % update rule for output weight
        Wi = Wi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        sumerr = sumerr + sum(er.^2);
    end
 end

% Validate the network.....................................................
rmstra = zeros(out,1);
res_tra = zeros(NTD,2);
for sa = 1: NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        rmstra = rmstra + (tt-Yo).^2;
        res_tra(sa,:) = [tt Yo];
end
disp(hid);
disp('Training Error');
disp(sqrt(rmstra/NTD));

% Test the network.........................................................
NFeature=load('approximationproblems\SI.tes');
Nout=load('approximationproblems\SI.y');
[NTD,~]=size(NFeature);
[NTD_out,~]=size(Nout);
rmstes = zeros(out,1);
res_tes = zeros(NTD,2);
for sa = 1: NTD
        xx = NFeature(sa,1:inp)';   % Current Sample
        ca = Nout(sa,1:1);      % Actual Output
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        rmstes = rmstes + (ca-Yo).^2;
        res_tes(sa,:) = [ca Yo];
end
disp('Testing Error');
disp(sqrt(rmstes/NTD));
