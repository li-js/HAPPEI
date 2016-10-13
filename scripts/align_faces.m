im_src='../data/det_crop_test/'

tar_src='../data/det_crop_test_aligned/'
mkdir(tar_src);

flist=dir([im_src '*.jpg']);

Len=length(flist);
for f=1:Len
    disp([f,Len])
    im_add=[im_src flist(f).name];
    pt_add=[im_add '_01.pts'];
    
    
    im=imread(im_add);
    if exist(pt_add,'file')==0
        continue
    else
        pts=importdata(pt_add);
        pts=pts.data;
    end
       
    M=pts;
    
    given_pos=[mean(pts(37:42,:));mean(pts(43:48,:));mean(pts(49:68,:))];
    eyepos = [-3,5;3,5;0,11];
    xrange = [-9,9];
    yrange = [0,18];
    scale = (xrange(2)-xrange(1))/223;
    tformAlign = cp2tform(given_pos ,eyepos,'nonreflective similarity' );
    img = imtransform( im,tformAlign,'bilinear', 'XData',xrange,'YData',yrange,'XYScale',scale );
        
    imwrite(img,[tar_src, flist(f).name]);
end