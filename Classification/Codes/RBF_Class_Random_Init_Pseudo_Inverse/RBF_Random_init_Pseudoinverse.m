clear all
close all
clc
Ntrain = load('Set 1/ION.tra');
[NTD,~] = size(Ntrain);
Nfeature = load('Set 1/ION.tes');

% Initialize the Algorithm Parameters.....................................
[~, tmp] = size(Ntrain(1, :));
inp = tmp - 1;                          % No. of input neurons
hid = 40;                               % No. of hidden neurons
out = max(Ntrain(:, inp+1)');           % No. of Output Neurons


% Train the network.......................................................
xx = randperm(size(Ntrain,1));
u = Ntrain(xx,: );
u = u(1:hid,1:inp);
sigma = zeros(hid,1);
dist = zeros(hid,hid);
x_train = Ntrain(:,1:inp);
y_train = Ntrain(:,inp+1);
x_tes = Nfeature(:,1:inp);
y_tes = load('Results/Group 1/ION.cla');
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
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(x_train(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
A = ones(NTD, out)*-1;
for i = 1 : NTD
    A(i, y_train(i))=1;
end
weights = pinv(phi)*A;
y_cross = phi * weights;
mis = 0;
valid_op = zeros(NTD, 1);
conf_cross = zeros(out, out);
for i = 1 : NTD
    t = find(y_cross(i,:) == max(y_cross(i, :)));
    valid_op(i) = t;
    conf_cross(y_train(i), t) = conf_cross(y_train(i), t) + 1;  
    if(t == y_train(i))
        mis = mis + 1;
    end
end
valid_op
no = 0;
ng = 1;
na = 0;
ni = 0;
for i = 1 : out
    no = no + conf_cross(i, i);
    ni = sum(conf_cross(i, :));
    na = na + conf_cross(i, i) / ni;
    ng = (100 * ng * conf_cross(i, i)) / ni;
end
no = (100 * no) / NTD
na = (100 * na) / out
ng = ng ^ (1/out)
conf_cross
[tot, ~] = size(Nfeature);
phi = zeros(tot, hid);
for i = 1 : tot
    for j = 1 : hid
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(x_tes(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
conf = zeros(out, out);
y_pred = phi * weights;
mis = 0;
op = zeros(tot, 1);
for i = 1 : tot
    t = find(y_pred(i,:) == max(y_pred(i, :)));
    op(i) = t;
    conf(y_tes(i), t) = conf(y_tes(i), t) + 1;  
    if(t == y_tes(i))
        mis = mis + 1;
    end
end
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
op
no = (100 * no) / tot
na = (100 * na) / out
ng = ng ^ (1/out)
conf
