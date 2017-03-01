clear all
close all
clc

hid1= [2:20];
[~,sizeHid] = size(hid1);
training_n=zeros(sizeHid,3);
testing_n=zeros(sizeHid,3);

trainingError=zeros(sizeHid);
testingError=zeros(sizeHid);
step = 1;
nos=zeros(19);
nas=zeros(19);
ngs=zeros(19);
for xxx=1:sizeHid

Ntrain = load('Wine.tra');
[NTD,~] = size(Ntrain);
Nfeature = load('Wine.tes');
Output = load('Wine.cla');

% Initialize the Algorithm Parameters.....................................
[~, tmp] = size(Ntrain(1, :));
inp = tmp - 1;                          % No. of input neurons
hid =hid1(xxx) ;                              % No. of hidden neurons
out = max(Ntrain(:, inp+1)');           % No. of Output Neurons
max_iterations = 10000;
%hid
% disp('Begin')
% Train the network.......................................................
ordering = randperm(NTD);

%K means clustering
inp_data = Ntrain(:,1:inp);
uinit = zeros(hid, size(inp_data, 2));
randidx = randperm(size(inp_data, 1));
uinit = inp_data(randidx(1:hid), :);
[u,memberships] = kMeans(inp_data,uinit,max_iterations);

qi = Nfeature(:,1:inp);
yo = Output(:,1);
sigma = zeros(hid,1);
dist = zeros(hid,hid);
y_train = Ntrain(:,inp+1);
for i = 1 : hid
    for j = 1 : hid
        dist(i,j) = sqrt(sum((u(j,:) - u(i,:)).^2));
    end
end
dmax = max(max(dist));
sigma = sigma + (dmax/sqrt(hid));
phi = zeros(NTD, hid);
for i = 1 : NTD
    for j = 1 : hid
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(Ntrain(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
A = ones(NTD, out)*-1;
for i = 1 : NTD
    A(i, y_train(i))=1;
end
weights = pinv(phi)*A;

conf_tr =zeros(out,out);
myo_tr=phi * weights;
c=0;
pre_tr = zeros(NTD,1);
for i = 1 : NTD
    ma = find(myo_tr(i,:) == max(myo_tr(i, :)));
     disp(ma)
    pre_tr(i) = ma;
    conf_tr(y_train(i), ma) = conf_tr(y_train(i), ma) + 1;  
    if(ma == y_train(i))
        c = c + 1;
    end
end
%pre_tr
% conf_tr
no_tr = 0;
ng_tr = 1;
na_tr = 0;
ni_tr = 0;
for i = 1 : out
    no_tr = no_tr + conf_tr(i, i);
    ni_tr = sum(conf_tr(i, :));
    na_tr = na_tr + conf_tr(i, i) / ni_tr;
    ng_tr = (100 * ng_tr * conf_tr(i, i)) / ni_tr;
end
no_tr = (100 * no_tr) / NTD;
na_tr = (100 * na_tr) / out;
ng_tr = nthroot(ng_tr, out);
training_n(hid,1)=no_tr;
training_n(hid,2)=na_tr;
training_n(hid,3)=ng_tr;

% disp('Test')
[tot, ~] = size(Nfeature);
phi2 = zeros(tot, hid);
for i = 1 : tot
    for j = 1 : hid
        phi2(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(qi(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
conf = zeros(out, out);
myo = phi2 * weights;
c = 0;
pre_op = zeros(tot, 1);
for i = 1 : tot
    ma = find(myo(i,:) == max(myo(i, :)));
     disp(ma)
    pre_op(i) = ma;
    conf(yo(i), ma) = conf(yo(i), ma) + 1;  
    if(ma == yo(i))
        c = c + 1;
    end
end
%pre_op
no = 0;
ng = 1;
na = 0;
ni = 0;
for i = 1 : out
    no = no + conf(i, i);
    ni = sum(conf(i, :));
    na = na + conf(i, i) / ni;
    ng = (100 * ng * conf(i, i)) / ni;
end
 conf_tr
 no_tr
 na_tr
 ng_tr
 conf
no = (100 * no) / tot
nos(step)=no;
na = (100 * na) / out
nas(step)=na;
ng = nthroot(ng, out)
ngs(step)=ng;
step=step+1;
testing_n(hid,1)=no;
testing_n(hid,2)=na;
testing_n(hid,3)=ng;
end
%disp('max')
%max(testing_n(:,1))
%max(testing_n(:,2))
%max(testing_n(:,3))
plot(hid1,nas,'g')
xlabel('Number of hidden neurons')
ylabel('Average accuracy')
title('Accuracy v/s number of hidden neurons')
% legend('Overall accuracy','Average accuracy','Geometric mean accuracy')
