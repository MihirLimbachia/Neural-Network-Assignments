% Program for Risk sensitive MLP..........................................

clear all
close all
clc
Ntrain=load('VC.tra');  % generate a 5x5 matrix random
[NTD,inp] = size(Ntrain);
out = 4;            % No. of Output Neurons
lam = 3.e-02;       % Learning rate
epo= 750;
inp=inp-1;
mat=[];
epoch=[];
mat1=[];
mat2=[];
count=0;
count1=0;
hid=30;
b=NTD;
%b=floor(b);% No. of hidden neurons
%for hid=5:2:20

Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  


% Initialize the weights..................................................


% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    miscla = 0;
    for sa = 1 : b
        xx = Ntrain(sa,1:inp)';     % Current Sample
        ti = Ntrain(sa,inp+1:end)';
        tt(1:out,1)=0;
        %disp(ti);
        tt(ti,1)=1;
        % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;
        
        Wo = Wo + lam * (er * Yh'); % update rule for output weight
        Wi = Wi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        sumerr = sumerr + sum(er.^2);
        ca = find(tt==1);           % actual class
        [~,cp] = max(Yo);           % Predicted class
        
    
  %disp([sumerr miscla])
%     save -ascii Wi.dat Wi;
%     save -ascii Wo.dat Wo;
end
end
% Validate the network.....................................................
conftra = zeros(out,out);
res_tra = zeros(b,2);
for sa = 1: b
        xx = Ntrain(sa,1:inp)';     % Current Sample
        ti = Ntrain(sa,inp+1:end)';
        tt(1:out,1)=-1;
        tt(ti,1)=1;% Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;
        %disp(Yo');
        %disp(tt');% Predicted output
        ca = find(tt==1);           % actual class
        [~,cp] = max(Yo);           % Predicted class
        conftra(ca,cp) = conftra(ca,cp) + 1;
        res_tra(sa,:) = [ca cp];
end

%disp(conftra);

for i=1 : out
    count1=count1+conftra(i,i);
end
%disp(count1/b);

mat = [mat;count1/b];
epoch = [epoch;ep];
count1=0;

NFeature=load('VC.tes');
[NTD,~]=size(NFeature);..............................
NResult=load('VC.cla');
conftes = zeros(out,out);
res_tes = zeros(NTD,1);
outmat=zeros(NTD,1);
for sa =1: NTD
        xx = NFeature(sa,1:inp)';   % Current Sample
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;
        aclass=NResult(sa,:);
        aclass=floor(aclass);
        [~,cp]=max(Yo);% Predicted output
        res_tes(sa,1)=cp;
        outmat(aclass)=outmat(aclass)+1;
        conftes(aclass,cp)=conftes(aclass,cp)+1;
        %disp(cp);           % Predicted class
        
end

%fid=fopen('VC.txt','w');
%fprintf(fid,'%d\n',out);
count=0;
countG=1;
countA=0;
inp=0;
for i=1 : out
   count=count+conftes(i,i);
%   if(outmat(i)~=0)
   countG=countG*conftes(i,i)/outmat(i);
   countA=countA + conftes(i,i)/outmat(i);
%    end
end
GA=100*nthroot(countG,out);
AA=countA*100/out;
OA=100*count/(NTD);
disp('hid')
disp(hid)
disp('epoch')
disp(epo)
disp('Overall')
disp(OA)
disp('Geometric')
disp(GA)
disp('Average')
disp(AA)

%mat1=[mat1;count/(NTD-b)];
%mat2=[mat2;hid];
%count=0;


%plot(epoch,mat);
