clc;
clear all;
close all;

setNo = '1';
str = 'VC';
Ntrain=load('C:\Users\Urmil\Desktop\study\SEM 6\neural network\rbf\sets\Set1\Wine.tra');
[NTD,~] = size(Ntrain);
[~, tmp] = size(Ntrain(1, :));
inp = tmp - 1;    
hid = 41;
out = max(Ntrain(:,end));
epo = 200;

Mu = zeros(hid,inp);    %centre  // will have to change for bipolar input
perm = randperm(NTD);
inp_data = Ntrain(:,1:inp);
uinit = kMeansInitCentroids(inp_data,hid);
[centres,memberships] = kMeans(inp_data,uinit,epo);
%get centres:
for i = 1:hid
   Mu(i,:) =  centres(i);
end

% now find sigma: dmax/sqrt(k)
d = dist(Mu');
dmax= max(d(:));
sig = zeros(hid,1);
for i=1:hid
    sig(i,1) = dmax/sqrt(hid);
end

lrSig = 1e-04;
lrCentre = 1e-03;
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
        pre_tr(sa) = ca;
        if ca~=cp 
            miscla_val = miscla_val + 1;
        end
        confusion(ca,cp) = confusion(ca,cp) + 1;
end
miscla_val
confusion

%Testing Network

test_inp = load('C:\Users\Urmil\Desktop\study\SEM 6\neural network\rbf\sets\Set1\Wine.tes');
opp = load('C:\Users\Urmil\Desktop\study\SEM 6\neural network\rbf\sets\Set1\WIne.cla');
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

overall_acc = (overall_acc * 100)/s
Geo_acc = (Geo_acc)^(1/out)
avg_acc = (avg_acc * 100) / out

        
        
        

