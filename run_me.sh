#! /bin/bash

#============Step 0: Gather Testing images and CENTRIST features=================
echo "Assume the testing data are placed under ./data/Test_images_distribute/"
echo "Assume the training and validation CENTRIST feature are under ./data/Train_Val_CENTRIST_Feat/"
echo "Assume the testing CENTRIST feature are under ./data/Test_CENTRIST_blocks_4_distribute/"

echo "The list of testing file is ./data/list_all_test.txt"
gunzip ./data/list_all_test.txt.gz


#============Step 1: Detect faces  ==========================
echo "Any face detector is OK. To reproduce the result, use the detection results at ./data/det_rects_test.txt"
gunzip ./data/det_rects_test.txt.gz

#============Step 2: Crop faces ==========================
cd scripts; matlab -nodesktop -nosplash -r "crop_det_test;  exit"; cd -; 


#============Step 3: Face landmark detection===============
echo "Any face landmark detector is OK. To reproduce the result, use the detection results at ./data/landmark_pts.gz"
tar -xvzf ./data/landmark_pts.gz -C ./data
cp ./data/landmark_pts/* ./data/det_crop_test/ -v


#============Step 4: Face Warpping ========================
cd scripts; matlab -nodesktop -nosplash -r "align_faces;  exit"; cd -; 


#============Step 5: Mix cropped and aligned faces ========
cd scripts; matlab -nodesktop -nosplash -r "generate_list;  exit"; cd -; 


#============Step 6: Get dimension reduced CENTRIST feature=====
cd scripts; matlab -nodesktop -nosplash -r "process_scene_feat_testing;  exit"; cd -; 


#============Step 7: Extract face feature from CNN=====
# install caffe in the current directory, copy scripts/feat_extract_to_file.cpp to ./caffe/tools/
git clone https://github.com/BVLC/caffe.git
cp scripts/feat_extract_to_file.cpp ./caffe/tools/ 
cd caffe; mkdir build; cd build; cmake ../; make -j; cd ../../; # make sure caffe builds well

# extract features with caffe models
./caffe/build/tools/feat_extract_to_file ./models/model1.caffemodel ./models/mem_test.prototxt pool5 \
data/list_det_crop_align_filled.txt ./data/ ./data/feat1.txt

./caffe/build/tools/feat_extract_to_file ./models/model2.caffemodel ./models/mem_test.prototxt pool5 \
data/list_det_crop_align_filled.txt ./data/ ./data/feat2.txt


#============Step 8: Test with LSTM====================
#install apollocaffe, instructions available at http://apollocaffe.com/
git clone http://github.com/Russell91/apollocaffe.git # make sure apollocaffe builds well 

python scripts/test_lstm.py --config models/config.json --gpu 0
python scripts/test_lstm_ord.py --config models/config.json --gpu 0
