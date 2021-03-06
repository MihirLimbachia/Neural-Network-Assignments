% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc
% Load the training data..................................................
Ntrain=load('approximationproblems\mg85.tra');
[NTD,~] = size(Ntrain);

% Initialize the Algorithm Parameters.....................................
inp = 4;            % No. of input neurons
hid = 11;                % No. of hidden neurons
out = 1;            % No. of Output Neurons
lam = 1.e-02;       % Learning rate
epo=1200;

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,inp);
    DWo = zeros(out,hid);
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        DWo = DWo + lam * (er * Yh'); % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        if(rem(sa,10)==0)
            Wi = Wi + DWi;
            Wo = Wo + DWo;
            DWi = zeros(hid,inp);
            DWo = zeros(out,hid);
        end
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
%         disp(sqrt(rmstra/NTD));    
end
disp(hid);
disp('Training Error');
disp(sqrt(rmstra/NTD));

% Test the network.........................................................
NFeature=load('approximationproblems\mg85.tes');
Nout=load('approximationproblems\mg85.y');
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

