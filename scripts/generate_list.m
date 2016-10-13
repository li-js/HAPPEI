
root='../data/';

src1='det_crop_test/';
src2='det_crop_test_aligned/';


add=[root 'list_det_crop_align_filled.txt'];
fid=fopen(add,'w');

flist=dir([root src1 '*.jpg']);

for f=1:length(flist)
	if  exist([root src2 flist(f).name])
		fprintf(fid, '%s%s %d\n', src2, flist(f).name, 0);
	else
		fprintf(fid, '%s%s %d\n', src1, flist(f).name, 0);
	end
end

fclose all;
disp(['Written to: ' add])