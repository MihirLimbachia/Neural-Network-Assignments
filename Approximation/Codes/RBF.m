clc;
clear all;
close all;

trainingError = 0;
testingError = 0;

Ntrain=load('approximationproblems/bj.tra');
[NTD,~] = size(Ntrain);

[~, tmp] = size(Ntrain(1, :));
inp = tmp - 1;      % No. of input neurons
hid = 4;            % No. of hidden neurons
out = 1;            % No. of Output Neurons
epo = 800;
lrw = 1e-02;        % Learning rate for weights
lrs = 1e-04;        % Learning rate for sigma
lrc = 1e-03;        % Learning rate for centres
u = zeros(hid,inp);             %centre  
xx = randperm(NTD);
centres = xx(1:hid);
for i = 1:hid
   u(i,:) = Ntrain(centres(i),1:inp);
end
d = dist(u');
dmax = max(d(:));
sig = zeros(hid,1);
for i=1:hid
    sig(i,1) = dmax/(sqrt(hid));
end 
for ep = 1:epo
    sumerr = 0;
    for sa = 1:NTD
        t = Ntrain(sa,1:inp)';
        xx = repmat(t',hid,1);
        tt = Ntrain(sa,inp+1:end)';
        if sa == 1&& ep == 1
              w = (pinv(exp(-sum((xx - u).^2,2)./(2*sig.^2))')*tt')';   
        end
        tmp = abs(xx - u);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
        er = tt - Yo;
        dW = er * phi';
        w = w + lrw * dW;
        tmp2 = (xx - u);
        tmp3 = bsxfun(@rdivide,tmp2,sig.^2);
        dC = bsxfun(@times,-2*(er'*w).*phi',tmp3');
        u = u - lrc * dC';
        tmp4 = bsxfun(@rdivide,tmp.^2,sig.^3);
        tmp5 = sum(tmp4,2);
        dS = -2*(er'*w).*phi'.*tmp5';
        sig = sig - lrs*dS';
        sumerr = sumerr + sum(er.^2);   
    end
    disp(sqrt(sumerr/NTD));
end
rmstra = zeros(out,1);
res_tra = zeros(NTD,2);
pre1 = zeros(1, NTD);
ac1 = zeros(1, NTD);
x1 = zeros(1, NTD);
for sa = 1: NTD
        x1(1, sa) = sa;
        t = Ntrain(sa,1:inp)';   % Current Sample
        xx = repmat(t',hid,1);
        tt = Ntrain(sa,end);      % Actual Output
        tmp = abs(xx - u);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
        pre1(1, sa) = Yo;
        ac1(1, sa) = tt;
        er = tt - Yo;
        rmstra = rmstra + sum(er.^2);
        res_tra(sa,:) = [tt Yo];
end
NTD1 = NTD;
disp('Training Error');
disp(sqrt(rmstra/NTD));
trainingError=(sqrt(rmstra/NTD));
NFeature=load('approximationproblems/bj.tes');
NFeature2=load('approximationproblems/bj.y');
[NTD,~]=size(NFeature);
rmstes = zeros(out,1);
res_tes = zeros(NTD,2);
pre2 = zeros(1, NTD);
ac2 = zeros(1, NTD);
x2 = zeros(1, NTD);
for sa = 1: NTD
        x2(1,sa) = sa;
        t = NFeature(sa,1:inp)';   % Current Sample
        xx = repmat(t',hid,1);
        tt = NFeature2(sa,end);      % Actual Output
        tmp = abs(xx - u);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
        er = tt - Yo;
        pre2(1, sa) = Yo;
        ac2(1, sa) = tt;
        rmstes = rmstes + sum(er.^2);
        res_tes(sa,:) = [tt Yo];
end
disp('Testing Error');
disp(sqrt(rmstes/NTD));
testingError=(sqrt(rmstes/NTD));
plot(x1, pre1, x1, ac1);
legend('Predicted Output', 'Actual Output');
figure;
plot(x2, pre2, x2, ac2);
legend('Predicted Output', 'Actual Output');