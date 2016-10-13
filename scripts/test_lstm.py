import numpy as np
import json
import itertools

import apollocaffe
from apollocaffe.layers import NumpyData, LstmUnit, Concat, InnerProduct, Filler

data_root='./data/'


def evaluate_forward(net, net_config, feat,scene_feat):
    net.clear_forward()
    feat_dim=feat.shape[1]

    net.f(NumpyData("prev_hidden", np.zeros((1, net_config["mem_cells"]))))
    net.f(NumpyData("prev_mem", np.zeros((1, net_config["mem_cells"]))))
    filler = Filler("uniform", net_config["init_range"])
    predictions = []

    length = feat.shape[0]+1
    for step in range(length):
        net.clear_forward()
        if step==0:
            value=scene_feat.reshape(1,feat_dim)
        else:
            value = feat[step-1,:].reshape(1,feat_dim)
        net.f(NumpyData("value", data=value ))
        prev_hidden = "prev_hidden"
        prev_mem = "prev_mem"
        net.f(Concat("lstm_concat", bottoms=[prev_hidden, "value"]))
        net.f(LstmUnit("lstm", net_config["mem_cells"],
            bottoms=["lstm_concat", prev_mem],
            param_names=[
                "input_value", "input_gate", "forget_gate", "output_gate"],
            weight_filler=filler,
            tops=["next_hidden", "next_mem"]))
        net.f(InnerProduct("ip", 1, bottoms=["next_hidden"]))
        predictions.append(float(net.blobs["ip"].data.flatten()[0]))
        net.blobs["prev_hidden"].data_tensor.copy_from(
            net.blobs["next_hidden"].data_tensor)
        net.blobs["prev_mem"].data_tensor.copy_from(
            net.blobs["next_mem"].data_tensor)
    return predictions

def evaluate(config, test_data):
    eval_net = apollocaffe.ApolloNet()
    net_config = config["net"]
    feat=test_data['feats'][0]
    feat2=test_data['feats2'][0]
    feat=np.hstack((feat,feat2))
    print feat.shape
    evaluate_forward(eval_net, net_config, feat, test_data['scene_feats'][0])
    net_add="%s_%d.h5" % (config["logging"]["snapshot_prefix"], config["solver"]["max_iter"] - 1)
    net_add='models/model_lstm.h5'
    eval_net.load(net_add)

    residuals=[]
    fid=open(config["logging"]["result_file"],'w');
    for idx in xrange(len(test_data['labels'])):
        feat=test_data['feats'][idx]
        feat2=test_data['feats2'][idx]
        feat=np.hstack((feat,feat2))
        scene_feat=test_data['scene_feats'][idx]
        label=test_data['labels'][idx]
        predictions=[]
        max_len=min(8, feat.shape[0])
        all_permu=list(itertools.permutations(range(max_len),max_len))
        some_permu=all_permu[0:min(10,len(all_permu))]
        for idx in some_permu:
            prediction = evaluate_forward(eval_net, net_config, feat[idx,:],scene_feat)
            prediction = prediction[-1]
            predictions.append(prediction)
        print 'File: %s, Predicted: %0.5f' % (label, np.array(predictions).mean()) 
        fid.write('%s %0.5f\n' % (label, np.array(predictions).mean()))
    print "%d images processed using %s" % (len(test_data['labels']),net_add)
    fid.close()

def main():
    parser = apollocaffe.base_parser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    list_add=data_root+'list_all_test.txt'
    list_crop_add=data_root+'list_det_crop_align_filled.txt'
    feat_add=data_root+'feat1.txt'

    train_gt=np.loadtxt(list_add, dtype={'names': ('name', 'label'), 'formats': ('S200', 'i4')})
    train_crop_gt=np.loadtxt(list_crop_add, dtype={'names': ('name', 'label'), 'formats': ('S200', 'i4')})
    train_feat=np.loadtxt(feat_add)

    train_feat_list=[]
    train_label_list=[]
    assert(len(train_crop_gt)==train_feat.shape[0])
    im_list=train_crop_gt['name'];
    for k in xrange(len(im_list)):
        im_list[k]=im_list[k].split('/')[-1][0:-7]
    im_list_uniq=list(set(im_list))
    for s in im_list_uniq:
        idx=s==train_crop_gt['name']
        feat=train_feat[idx,:]
        train_feat_list.append(feat)
        train_label_list.append(s)

    feat_add2=data_root+'feat2.txt'

    train_feat2=np.loadtxt(feat_add2)
    train_feat_list2=[]
    assert(len(train_crop_gt)==train_feat2.shape[0])
    for s in im_list_uniq:
        idx=s==im_list
        feat2=train_feat2[idx,:]
        train_feat_list2.append(feat2)

    holistic_feat_add=data_root+'feat_centrist_test_d1024.txt'
    holistic_feat=np.loadtxt(holistic_feat_add)
    scene_feat_list=[]
    for k,s in enumerate(im_list_uniq):
        idx=s.split('/')[-1]==train_gt['name']
        assert(idx.sum()==1)
        scene_feat_list.append(holistic_feat[idx,:])


    test_data={'feats': train_feat_list, 'feats2': train_feat_list2, 'labels': train_label_list, 'scene_feats':  scene_feat_list, 'current_idx': 0}

    evaluate(config, test_data)

if __name__ == "__main__":
    main()
