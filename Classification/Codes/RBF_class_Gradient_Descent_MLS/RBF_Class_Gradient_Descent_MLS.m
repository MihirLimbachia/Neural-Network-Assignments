% Program for MLP..........................................
% Update weights for a given epoch

%function op1 = RBF_MLS(set_no, file_name, EPO, INP, HID, OUT,LAM)
    set_no='1';
file_name = 'ae';
EPO='1000';
INP='5';
HID='6';
OUT='4';
LAM='0.01';
minAA=0;
minhid=0;
minepo=0;
minGA=0;
minOA=0;
setNo=set_no;
str=file_name;
s1=strcat('Assignment Classification\Set',setNo,'\',str,'.tra');
s2=strcat('Assignment Classification\Set',setNo,'\',str,'.tes');
s3=strcat('Assignment Classification\Results\Group',setNo,'\',str,'.cla');
Ntrain=load(s1);
test_inp = load(s2);
opp = load(s3);
[NTD,inp] = size(Ntrain);
inp = str2num(INP); % No. of input neurons
hid = str2num(HID); % No. of hidden neurons
out = str2num(OUT); % No. of Output Neurons
lam = str2double(LAM);% Learning rate
epo = str2num(EPO);

Mu = zeros(hid,inp);    %centre  // will have to change for bipolar input
perm = randperm(NTD);
centres = perm(1:hid);
%get centres:
for i = 1:hid
Mu(i,:) =  Ntrain(centres(i),1:inp);
end

% now find sigma: dmax/sqrt(k)
d = dist(Mu');
dmax= max(d(:));
sig = zeros(hid,1);
for i=1:hid
sig(i,1) = dmax/sqrt(hid);
end

lrSig = 1e-4;
lrCentre = 1e-3;
lrWeight = 1e-02;
%[trI,valI,testI] = dividerand(NTD,.9,.1,0);
%[~,NumTr] = size(trI);
%[~,NumVal] = size(valI);

for iter = 1:epo
sumerr = 0;
miscla = 0;
for sa = 1:NTD
%input_index = trI(1,sa);
x = Ntrain(sa,1:inp)';
xx = repmat(x',hid,1);
tt=zeros(1,out);
class = Ntrain(sa,end);
for i = 1:out
if i == class
tt(1,i)=1;
else
tt(1,i)=-1;
end
end
tt=tt';

if sa == 1&& iter==1
w = (pinv(exp(-sum((xx - Mu).^2,2)./(2*sig.^2))')*tt')';   %out x hid
end
tmp = abs(xx - Mu);
tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
phi = exp(-sum(tmp1,2));
Yo = w*phi;
er = tt - Yo;
extra=tt.*Yo;
for i=1:out
if(extra(i)>1)
er(i)=0;
end
end;
deltaW = er * phi';
w = w + lrWeight * deltaW;

tmp2 = (xx - Mu);
tmp3 = bsxfun(@rdivide,tmp2,sig.^2);
deltaC = bsxfun(@times,-2*(er'*w).*phi',tmp3');
Mu = Mu - lrCentre * deltaC';

tmp4 = bsxfun(@rdivide,tmp.^2,sig.^3);
tmp5 = sum(tmp4,2);
deltaSigma = -2*(er'*w).*phi'.*tmp5';
sig = sig - lrSig*deltaSigma';


sumerr = sumerr + sum(er.^2);
ca = find(tt==1);           % actual class
[~,cp] = max(Yo);           % Predicted class
if ca~=cp 
miscla = miscla + 1;
end
end
if(rem(iter,150)==0)
iter
end
end

confusion = zeros(out,out);
miscla_val = 0;
pre_tr = zeros(NTD,1);
for sa = 1 : NTD
% input_index = valI(1,sa);
x = Ntrain(sa,1:inp)';
xx = repmat(x',hid,1);
tt=zeros(1,out);
class = Ntrain(sa,end);
for i = 1:out
if i == class
tt(1,i)=1;
else
tt(1,i)=-1;
end
end
tt=tt';

if sa == 1&& iter==1
w = (pinv(exp(-sum((xx - Mu).^2,2)./(2*sig.^2))')*tt')';   %out x hid
end
tmp = abs(xx - Mu);
tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
phi = exp(-sum(tmp1,2));
Yo = w*phi;
er = tt - Yo;

sumerr = sumerr + sum(er.^2);
ca = find(tt==1);           % actual class
[~,cp] = max(Yo);           % Predicted class
pre_tr(sa) = cp;
if ca~=cp 
miscla_val = miscla_val + 1;
end
confusion(ca,cp) = confusion(ca,cp) + 1;
end
miscla_val
confusion

%Testing Network


[s,~] = size(test_inp);
miscla_test = 0;
confusion_test = zeros(out,out);
overall_acc = 0;
Geo_acc = 1;
avg_acc = 0;
rmserr = 0;
f_op = zeros(s);
pre_op = zeros(s,1);
for sa = 1 : s
%input_index = valI(1,sa);
x = test_inp(sa,1:inp)';
xx = repmat(x',hid,1);
tt=zeros(1,out);
class = opp(sa,1);
for i = 1:out
if i == class
tt(1,i)=1;
else
tt(1,i)=-1;
end
end
tt=tt';

if sa == 1&& iter==1
w = (pinv(exp(-sum((xx - Mu).^2,2)./(2*sig.^2))')*tt')';   %out x hid
end
tmp = abs(xx - Mu);
tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
phi = exp(-sum(tmp1,2));
Yo = w*phi;
er = tt - Yo;

rmserr = rmserr + sum(er.^2);
ca = find(tt==1);           % actual class
[~,cp] = max(Yo);           % Predicted class
pre_op(sa) = cp ;
if ca~=cp 
miscla_test = miscla_test + 1;
end
f_op(sa) = cp;
confusion_test(ca,cp) = confusion_test(ca,cp) + 1;
end
for var = 1 : out
Ni = sum(confusion_test(var,:));
overall_acc = overall_acc + confusion_test(var,var);
Geo_acc = Geo_acc * ((100*confusion_test(var,var))/Ni);
avg_acc = avg_acc + (confusion_test(var,var) / Ni);
end
confusion_test
miscla_test

OA = (overall_acc * 100)/s
GA = (Geo_acc)^(1/out)
AA = (avg_acc * 100) / out
