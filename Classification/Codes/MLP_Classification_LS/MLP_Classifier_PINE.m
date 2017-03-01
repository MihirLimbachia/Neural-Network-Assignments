% Program for MLP..........................................
% Update weights for a given epoch

clear all
close all
clc

maxOA=-1;
maxhid=-1;
maxepo=-1;
%for hid1 = 5 : 20
%for epo1 = 100 : 50 : 1500
% Load the training data..................................................
Ntrain=load('PIMA.tra');
[NTD,inp] = size(Ntrain);
Ntrain=Ntrain(randperm(size(Ntrain,1)),:);

%NTD1=NTD;

%NTD=floor(0.9*NTD1);
% Initialize the Algorithm Parameters.....................................
inp = inp-1; % No. of input neurons
hid = 10; % No. of hidden neurons
%hid=hid1;
out = 2; % No. of Output Neurons
lam = 1.e-03; % Learning rate
epo = 800;
%epo=epo1;

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0); % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0); % Output weights

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,inp);
    DWo = zeros(out,hid);
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)'; % Current Sample
        cno = Ntrain(sa,inp+1:end)'; % Current Target
        tt(1:out,1)=0;
        %disp(ti);
        tt(cno,1)=1;
        Yh = 1./(1+exp(-Wi*xx)); % Hidden output
        Yo = Wo*Yh; % Predicted output
        er = tt - Yo; % Error
        DWo = DWo + lam * (er * Yh'); % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx'; %update for input weight
        sumerr = sumerr + sum(er.^2);
    end
    Wi = Wi + DWi;
    Wo = Wo + DWo;
    %disp(sqrt(sumerr/NTD))
end

%NTD=NTD1;
% Validate the network.....................................................
Ntrain=load('PIMA.tra');
[NTD,~] = size(Ntrain);

val_conf=zeros(out,out);
rmstra = zeros(out,1);
res_tra = zeros(NTD,1);
for sa = 1: NTD
    xx = Ntrain(sa,1:inp)'; % Current Sample
    tt = Ntrain(sa,inp+1:end)'; % Current Target
    Yh = 1./(1+exp(-Wi*xx)); % Hidden output
    Yo = Wo*Yh; % Predicted output
    rmstra = rmstra + (tt-Yo).^2;
    aclass=round(tt);
    [~,pclass] = max(Yo);% Predicted class
    %disp(pclass)
    %disp(aclass)
    val_conf(aclass,pclass) = val_conf(aclass,pclass) + 1;
    res_tra(sa,:) = [pclass];
    %disp(tt)
    %disp(Yo)
end
%disp(val_conf)
%disp(sqrt(rmstra/NTD))

for sa = 1 : NTD
    %disp(res_tra(sa));
end

% Test the network.........................................................
Test_conf=zeros(out,out);
NFeature=load('PIMA.tes');
NResult=load('PIMA.cla');
[NTD,~]=size(NFeature);
rmstes = zeros(out,1);
res_tes = zeros(NTD,1);
outmat=zeros(out);
for sa = 1: NTD
    xx = NFeature(sa,1:inp)'; % Current Sample
    ca = NResult(sa,end); % Actual Output
    Yh = 1./(1+exp(-Wi*xx)); % Hidden output
    Yo = Wo*Yh; % Predicted output
    rmstes = rmstes + (ca-Yo).^2;
    %res_tes(sa,:) = [ca Yo];
    aclass=round(ca);
    outmat(aclass)=outmat(aclass)+1;
    [~,pclass] = max(Yo); % Predicted class
    Test_conf(aclass,pclass) = Test_conf(aclass,pclass) + 1;
    res_tes(sa,:) = [pclass];
    %disp(res_tes(sa,:));
end
%disp(sqrt(rmstes/NTD))
%}
%disp(Test_conf)

for sa = 1 : NTD
    %   disp(res_tes(sa));
end

count=0;
countG=1;
countA=0;
nz=0;
for i=1 : out
    count=count+Test_conf(i,i);
    % if(outmat(i)~=0)
    countG=countG*Test_conf(i,i)/outmat(i);
    countA=countA + Test_conf(i,i)/outmat(i);
    nz=nz+1;
    % end
end
GA=100*nthroot(countG,nz);
AA=countA*100/nz;
OA=count*100/(NTD);

OA
%GA
%AA
%end
GA
AA