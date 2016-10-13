
src='../data/list_all_test.txt';
det_add='../data/det_rects_test.txt';
im_src='../data/Test_images_distribute/';

tar_src= '../data/det_crop_test/';
mkdir(tar_src)
im_list=importdata(src);

det_rects=importdata(det_add);


Min_sz=34;
Conf_thres=0.98;

for f=1:length(im_list)
    disp([f,length(im_list)])
    tmp=strsplit(im_list{f}, '/');
    key=tmp{end};
    map=strcmp(key,det_rects.textdata);
    conf=det_rects.data(map,1);
    rects=det_rects.data(map,2:5);
    rects(:,3:4)=rects(:,3:4)-rects(:,1:2);
    

    im_add=[im_src key];
    
    im=imread(im_add);
    im_sz=size(im);
    
    idx=conf>Conf_thres;
    conf=conf(idx);
    rects=rects(idx,:);
    
    for ii=1:length(conf)
        rect2=rectify_rect(rects(ii,:),im_sz(1:2),1.2);
        if rect2(4)<Min_sz
            continue
        end
        
        im3=imcrop(im,rect2);        
        imwrite(im3,sprintf('%s/%s_%02d.jpg', tar_src,key, ii));
    end

end


key='3121781_150da6c168_z.jpg'   % this testing image has no face fulliing the above criterion. We take one face from the detection result
map=strcmp(key,det_rects.textdata);
conf=det_rects.data(map,1);
rects=det_rects.data(map,2:5);
rects=rects(1,:);
rects(:,3:4)=rects(:,3:4)-rects(:,1:2);
im_add=[im_src key];
im=imread(im_add);
rect2=rectify_rect(rects(1,:),im_sz(1:2),1.2);
im3=imcrop(im,rect2);        
imwrite(im3,sprintf('%s/%s_%02d.jpg', tar_src,key, ii));

