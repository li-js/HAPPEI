#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "caffe/layers/memory_data_layer.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>




using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

using boost::shared_ptr;

using namespace std;
using namespace cv;



template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);


int main(int argc, char** argv) {
    if (argc!=7)
    {
      cout<<"Wrong number of arguments. Usage: "<<argv[0]<<"input_caffemodel input_prototxt blob_name image_add_list image_root result_add"<<endl;
      exit(1);
    }

    string pretrained_binary_proto=argv[1];
    string feature_extraction_proto=argv[2];
    string extract_feature_blob_name=argv[3];
    string img_list=argv[4];
    string img_root=argv[5];
    string result_add=argv[6];
    bool use_gpu=true;
    if (use_gpu){ 
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);
    }
    else
    Caffe::set_mode(Caffe::CPU);    

    boost::shared_ptr<Net<float> > feature_extraction_net(new Net<float>(feature_extraction_proto, caffe::TEST));
    
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);    
  

  CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
        << "Unknown feature blob name " << extract_feature_blob_name
        << " in the network " << feature_extraction_proto;




  ifstream fin(img_list.c_str());
  ofstream fout(result_add.c_str());
  string imgFileName;
  int label_gt;

  int h_off = 0;
  int w_off = 0;
  int crop_size=224;
  bool resize_flag=true;
  int new_size=256;

  LOG(ERROR)<< "Loading model from: "<<pretrained_binary_proto;
  if (resize_flag) LOG(ERROR)<< "Resizing image to: "<<new_size;
  LOG(ERROR)<< "Cropping image to: "<<crop_size;
  LOG(ERROR)<< "Extacting Features to "<<result_add;


  while (fin>>imgFileName>>label_gt)
  {
    LOG(INFO)<<"Processing image: "<<img_root+imgFileName;
    Mat cv_img = imread(img_root+imgFileName);
    
    if (resize_flag){
        cv::Size cv_new_size(new_size,new_size);
        cv::resize(cv_img, cv_img,cv_new_size, 0, 0, cv::INTER_LINEAR);
      }

    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;
    h_off = (img_height - crop_size) / 2;
    w_off = (img_width - crop_size) / 2;
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    Mat cv_cropped_img = cv_img(roi);

    vector<Blob<float>*> input_vec;
    vector<Mat> dv;
    dv.push_back(cv_cropped_img);
    vector<int> dvl;
    dvl.push_back(0);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float> >(feature_extraction_net->layers()[0])->AddMatVector(dv,dvl);
    feature_extraction_net->Forward(input_vec);
    const boost::shared_ptr<Blob<float> > feature_blob = feature_extraction_net
          ->blob_by_name(extract_feature_blob_name);
    int batch_size = feature_blob->num();
    CHECK_EQ(batch_size,1) << "Batch size should be 1.";
    int dim_features = feature_blob->count() / batch_size;
    LOG(INFO) <<"Feature dim is "<<dim_features;
    const float* feature_blob_data = feature_blob->cpu_data();

   
    LOG(INFO) <<imgFileName<<" "<<feature_blob_data[0]<<" "<<label_gt;

    
    for (int k=0; k<dim_features; k++)
        fout <<feature_blob_data[k]<<" ";
    fout<<endl;

}
  LOG(ERROR)<< "Successfully extracted the features!";

 
  
  return 0;
}
