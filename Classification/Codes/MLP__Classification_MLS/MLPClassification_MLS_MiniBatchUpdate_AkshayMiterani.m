% Code Author : Akshay Miterani

% Program for MLP..........................................
% Update weights for a given epoch

function op = run_with_ease(set_no, file_name, EPO, INP, HID, OUT)
op_file=strjoin({"Outputs/",file_name,"/Output_",file_name,set_no},'');
fid = fopen (op_file, "w");

fprintf(fid, "\t\tRunning for %s - %s\n\n",file_name,set_no);
minAA=0;
minhid=0;
minepo=0;
minGA=0;
minOA=0;
setNo=set_no;
str=file_name;
s1=strjoin({"/Users/akshaymiterani/Downloads/Assignment Classification/Set ",setNo,"/",str,".tra"},'');
s2=strjoin({"/Users/akshaymiterani/Downloads/Assignment Classification/Set ",setNo,"/",str,".tes"},'');
s3=strjoin({"/Users/akshaymiterani/Downloads/Assignment Classification/Results/Group ",setNo,"/",str,".cla"},'');
Ntrain=load(s1);
NFeature=load(s2);
NResult=load(s3);
[NTD,inp] = size(Ntrain);

% Initialize the Algorithm Parameters.....................................
inp = str2num(INP); % No. of input neurons
hid = str2num(HID); % No. of hidden neurons
out = str2num(OUT); % No. of Output Neurons
lam = 1e-02; % Learning rate
epo = str2num(EPO);
gamma = 0.9;
% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0); % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0); % Output weights

Btaches = int32(inp/3);
Mini_Batch_Size = int32(inp/Btaches);
Mini_Batch_Size
% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,inp);
    DWo = zeros(out,hid);
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)'; % Current Sample
        cno = Ntrain(sa,inp+1:end)'; % Current Target
        tt(1:out,1)=-1;
        %disp(ti);
        tt(cno,1)=1;
        Yh = 1./(1+exp(-Wi*xx)); % Hidden output
        Yo = Wo*Yh; % Predicted output
        %er = zeros(out,1); % Error
        er=tt-Yo;
        extra=tt.*Yo;
        for i=1:out
            if(extra(i)>1)
                er(i)=0;
            end
        end;
        DWo = DWo + lam * (er * Yh'); % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx'; %update for input weight
        sumerr = sumerr + sum(er.^2);
        b = mod(int32(sa) ,int32(Mini_Batch_Size));
        if(b==0)
          Wi = Wi + DWi;
          Wo = Wo + DWo;
          DWi = zeros(hid,inp);
          DWo = zeros(out,hid);
        end
    end
    Wi = Wi + DWi;
    Wo = Wo + DWo;
%disp(sqrt(sumerr/NTD))
end

% Validate the network.....................................................
val_conf=zeros(out,out);
rmstra = zeros(out,1);
res_tra = zeros(NTD,2);
for sa = 1:NTD
        xx = Ntrain(sa,1:inp)'; % Current Sample
        cno = Ntrain(sa,inp+1:end)'; % Current Target
%       tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx)); % Hidden output
        Yo = Wo*Yh; % Predicted output
        rmstra = rmstra + (tt-Yo).^2;
        aclass=round(cno);
        [~,pclass] = max(Yo);% Predicted class
        %disp(pclass)
        %disp(aclass)
        val_conf(aclass,pclass) = val_conf(aclass,pclass) + 1;
        res_tra(sa,:) = [aclass pclass];
        %disp(tt)
        %disp(Yo)
end
fprintf(fid, '%d:%d:%d\n',inp,hid,out);
fprintf(fid, 'Epoch -> %d\n',epo);
fdisp(fid, val_conf);
fprintf("\n");
TrainNTD=NTD;
% Test the network.........................................................
Test_conf=zeros(out,out);
[NTD,~]=size(NFeature);
rmstes = zeros(out,1);
res_tes = zeros(NTD,2);
outmat=zeros(out);
for sa = 1: NTD
        xx = NFeature(sa,1:inp)'; % Current Sample
        ca = NResult(sa,end); % Actual Output
        Yh = 1./(1+exp(-Wi*xx)); % Hidden output
        tt(1:out,1)=-1; 
        tt(ca,1)=1; 
        Yo = Wo*Yh; % Predicted output
%         er = zeros(out,1); % Error
%         er=tt.*Yo;
%         for i=1:out
%             if(er(i)>1)
%                 er(i)=0;
%             end
%         end;
        rmstes = rmstes + (ca-Yo).^2;
        %res_tes(sa,:) = [ca Yo];
        aclass=round(ca);
        outmat(aclass)=outmat(aclass)+1;
        [~,pclass] = max(Yo); % Predicted class
        Test_conf(aclass,pclass) = Test_conf(aclass,pclass) + 1;
        res_tes(sa,:) = [aclass pclass];
        %disp(res_tes(sa,:));
end
fdisp(fid, Test_conf);

count=0;
countG=1;
countA=0;
nz=0;
for i=1 : out
    count=count+Test_conf(i,i);
    countG=countG*Test_conf(i,i)/outmat(i);
    countA=countA + Test_conf(i,i)/outmat(i);
    nz=nz+1;
end
GA=100*nthroot(countG,nz);
AA=countA*100/nz;
OA=100*count/(NTD);
fprintf(fid, "Overall Accuracy =");
fdisp(fid, OA)
fprintf(fid, "Geometric Mean Accuracy =");
fdisp(fid, GA)
fprintf(fid, "Average Accuracy =");
fdisp(fid, AA)
fprintf(fid,"\n\n");
fdisp(fid,'Train Result');
fdisp(fid, res_tra(1:TrainNTD,2));
fprintf(fid,"\n\n");
fdisp(fid,'Test Result');
fdisp(fid, res_tes(1:NTD,2));
fprintf(fid, "------------------------------------\n");
fclose (fid);
endfunction