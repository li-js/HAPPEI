src='../data/Train_Val_CENTRIST_Feat/'
src_test='../data/Test_CENTRIST_blocks_4_distribute/'


flist=dir([src '*.mat']);

Len=length(flist);

dim=4064;
T_dim=1024;

data_all=zeros(Len,dim);

im_list=cell(Len,1);
for f=1:Len
    disp([f,Len])
    feat=load([src flist(f).name]);
    data_all(f,:)=feat.var;
    im_list{f}=strrep(flist(f).name,'.mat', '');
end

data_mean=mean(data_all);

data_all=data_all-ones(Len,1)*data_mean;


Cov=data_all'*data_all;
[T,S, U]=svd(Cov);

Trans=T(:,1:T_dim);

data_dr=data_all*Trans;


data=importdata('../data/list_all_test.txt');
Len=length(data);
feat=zeros(Len,dim);
for f=1:Lenpre
    disp([f,Len])
    key=strrep(data{f},'.jpg', '.mat');
    key=strrep(key,'jpeg', '.mat');
    key=strrep(key,'.png', '.mat');
    xx=load([src_test key]);
    feat(f,:)=xx.var;
end

feat=feat-ones(Len,1)*data_mean;
feat_dr=feat*Trans;

dlmwrite('../data/feat_centrist_test_d1024.txt', feat_dr, ' ');



