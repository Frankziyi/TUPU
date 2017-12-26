import numpy as np
import os
import get_image_and_id_list
import model_InceptionV3
from scipy.misc import imread  # (h, w, c)
from scipy.misc import imresize

model_dir = "inception_v3.ckpt"
dataset_path = '../../world/data-gpu-94/sysu-reid/person-reid-data/Market-1501/Market-1501-v15.09.15/'
trainset_path = dataset_path + '/bounding_box_train/' # 751 id
gallery_path = dataset_path + '/bounding_box_test/' # 750 id
probe_path = dataset_path + '/query/'
save_path = 'features/'

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

def main():
    # set GPU and load trained model 
    inception_v3 = model_InceptionV3.pre_trained_inception_v3(model_dir)
    
    # dname = args.dataset
    # print '='*50
    # print dname
    
    # d = Dataset(join(root, dname))

    PX_name, PY = get_image_and_id_list.get_image_and_id_list(probe_path)
    GX_name, GY = get_image_and_id_list.get_image_and_id_list(gallery_path)

    PX = []

    for i in range(len(PX_name)):
        img_tmp = imread(probe_path + PX_name[i])
        img_tmp = img_tmp / 255.0
        img_tmp = imresize(img_tmp, (299, 299))
        img_tmp = (img_tmp - 0.5) * 2
        PX.append(img_tmp)
    PX = np.array(PX, dtype='f')
    print PX.shape

    GX1 = []

    for i in range(0, (1 + len(GX_name)) / 2):
        img_tmp = imread(gallery_path + GX_name[i])
        img_tmp = img_tmp / 255.0
        img_tmp = imresize(img_tmp, (299, 299))
        img_tmp = (img_tmp - 0.5) * 2
        GX1.append(img_tmp)
    GX1 = np.array(GX1, dtype='f')

    GX2 = []

    for i in range((1 + len(GX_name)) / 2, len(GX_name)):
        img_tmp = imread(gallery_path + GX_name[i])
        img_tmp = img_tmp / 255.0
        img_tmp = imresize(img_tmp, (299, 299))
        img_tmp = (img_tmp - 0.5) * 2
        GX2.append(img_tmp)
    GX2 = np.array(GX2, dtype='f')

    # print GX.shape

    
    # PX, PY, probc = d.get_test_probe(cams_list=[0])
    # GX, GY, galc = d.get_test_gallery(cams_list=[1])
    
    PX = inception_v3.get_feature(PX) # Batch_norm()( np.array(map(cropper, PX)) )
    GX1 = inception_v3.get_feature(GX1) # Batch_norm()( np.array(map(cropper, GX)) )
    GX2 = inception_v3.get_feature(GX2)

    GX = np.append(GX1, GX2, axis=0)

    print PX.shape
    print GX1.shape
    print GX2.shape
    
    np.save(save_path + 'test_probe_features.npy',PX)
    np.save(save_path + 'test_probe_labels.npy',PY)
    np.save(save_path + 'test_gallery_features.npy',GX)
    # np.save(save_path + 'test_gallery_features.npy',GX2)
    np.save(save_path + 'test_gallery_labels.npy',GY)
    print 'Done'
    
    # print 'normal cmc'
    # M = np.eye(PX.shape[1])
    # C = _eval_cmc(PX, PY, GX, GY, args.method)
   
    # file_suffix = args.method
    # np.save(osp.join(args.result_dir, 'cmc_' + file_suffix), C)
    # for topk in [1, 5, 10, 20]:
        # print "{:8}{:8.1%}".format('top-' + str(topk), C[topk - 1])
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('result_dir',
    #         help="Result directory. Containing extracted features and labels. "
    #              "CMC curve will also be saved to this directory.")
    # parser.add_argument('--model_dir',
    #                     help='Resnet model directory',
    #                     default = '/home/share/jiening/resnet/saves/grid/grid')
    # parser.add_argument('--dataset', type=str, 
    #                     help = 'the name of the datasets',
    #                     default = 'grid')
    # # parser.add_argument('--method', choices=['euclidean', 'cosine'],
    #         # default='euclidean')
    # parser.add_argument('--gpu', type = str, 
    #         default = '0')
    # args = parser.parse_args()
    main()      
                        

