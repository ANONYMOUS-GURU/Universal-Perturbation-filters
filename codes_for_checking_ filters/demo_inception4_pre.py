import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg,preprocess_image_batch3
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


from universal_pert import universal_perturbation
device = '/cpu:0'
num_classes = 10

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

if __name__ == '__main__':

    # Parse arguments
    argv = sys.argv[1:]

    # Default values
    path_train_imagenet = '/datasets2/ILSVRC2012/train'
    path_test_image = 'ILSVRC'
    
    try:
        opts, args = getopt.getopt(argv,"i:t:mx:my:sx:sy:kg:ka:",["test_image=","training_path=","mean_x=","mean_y=","std_x=","std_y=","k_filt=","k_gaus="])
    except getopt.GetoptError:
        print ('python ' + sys.argv[0] + ' -i <test image> -t <imagenet training path> -k <ksize>')
        sys.exit(2)
    
    mean_x=0.0
    mean_y=0.0
    std_x=1.0
    std_y=1.0
    kg=5
    ka=5

    for opt, arg in opts:
        if opt == '-t':
            path_train_imagenet = arg
        if opt == '-i':
            path_test_image = arg
        if opt=='-mx':
            mean_x=float32(arg)
        if opt=='-sx':
            std_x=float32(arg)
        if opt=='-my':
            mean_y=float32(arg)
        if opt=='-sy':
            std_y=float32(arg)
        if opt=='-kg':
            kg=int(arg)
        if opt=='-ka':
            ka=int(arg)

    with tf.device(device):
        persisted_sess = tf.Session()
        inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')

        if os.path.isfile(inception_model_path) == 0:
            print('Downloading Inception model...')
            urlretrieve ("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip", os.path.join('data', 'inception5h.zip'))
            # Unzipping the file
            zip_ref = zipfile.ZipFile(os.path.join('data', 'inception5h.zip'), 'r')
            zip_ref.extract('tensorflow_inception_graph.pb', 'data')
            zip_ref.close()

        model = os.path.join(inception_model_path)

        # Load the Inception model
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        persisted_sess.graph.get_operations()

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")

        print('>> Computing feedforward function...')
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))})

        def avg(img,k=5,s=1): return persisted_sess.run(tf.nn.avg_pool(img,ksize=(1,k,k,1),strides=(1,s,s,1),padding='SAME'))

        def percent_out(img): return np.round(np.exp(np.max(np.ravel(f(img))))/np.sum(np.exp(np.ravel(f(img))),axis=0)*100,decimals=3)

        def gaussian_kernel(size,meanx,meany,stdx,stdy):
            dx = (tf.distributions.Normal(meanx, stdx))
            dy = (tf.distributions.Normal(meany, stdy))
            valsx = dx.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
            valsy = dy.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
            gauss_kernel = tf.einsum('i,j->ij',valsx,valsy)
            return gauss_kernel / (tf.reduce_sum(tf.einsum('i,j->ij',valsx,valsy)))


        def do_gaus_conv(image,size=5,mean_x=0.0,std_x=1.0,mean_y=0.0,std_y=1.0):# img_original,mean_x,std_x,mean_y,std_y
            gauss_kernel = gaussian_kernel((size-1)/2,mean_x,mean_y,std_x,std_y)
            gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
            image1=np.reshape(image[:,:,:,0],[-1,image.shape[1],image.shape[2],1])
            image2=np.reshape(image[:,:,:,1],[-1,image.shape[1],image.shape[2],1])
            image3=np.reshape(image[:,:,:,2],[-1,image.shape[1],image.shape[2],1])
            cn1=tf.reshape((tf.nn.conv2d(image1, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")),[image.shape[1],image.shape[2]])
            cn2=tf.reshape((tf.nn.conv2d(image2, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")),[image.shape[1],image.shape[2]])
            cn3=tf.reshape((tf.nn.conv2d(image3, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")),[image.shape[1],image.shape[2]])
            return persisted_sess.run(tf.reshape(tf.stack((cn1,cn2,cn3),axis=2),[1,image.shape[1],image.shape[2],3]))
    


        file_perturbation = os.path.join('data', 'universal.npy')

        if os.path.isfile(file_perturbation) == 0:

            # TODO: Optimize this construction part!
            #print('>> Compiling the gradient tensorflow functions. This might take some time...')
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.placeholder(tf.int32, shape=(num_classes,))
            dydx = jacobian(y_flat,persisted_input,inds)

            print('>> Computing gradient function...')
            def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

            # Load/Create data
            datafile = os.path.join('data', 'imagenet_data.npy')
            if os.path.isfile(datafile) == 0:
                print('>> Creating pre-processed imagenet data...')
                X = create_imagenet_npy(path_train_imagenet)

                print('>> Saving the pre-processed imagenet data')
                if not os.path.exists('data'):
                    os.makedirs('data')

                # Save the pre-processed images
                # Caution: This can take take a lot of space. Comment this part to discard saving.
                np.save(os.path.join('data', 'imagenet_data.npy'), X)

            else:
                #print('>> Pre-processed imagenet data detected')
                X = np.load(datafile)

            # Running universal perturbation
            v = universal_perturbation(X, f, grad_fs, delta=0.2,num_classes=num_classes)

            # Saving the universal perturbation
            np.save(os.path.join(file_perturbation), v)

        else:
            #print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        #print('>> Testing the universal perturbation on an image')

        # Test the perturbation on the image
        labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')
        k=0
        dirs=os.listdir(path_test_image)
        final_list=[]
        for folder in dirs:
            
            print(' >> predicting for class {}'.format(folder))
            path=os.path.join(path_test_image,folder)

            image_original = preprocess_image_batch3(path, img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")
            print(' >> has {} images'.format(image_original.shape[0]))

            list_out=[]
            names_images=os.listdir(path)
            
            for x in range(image_original.shape[0]):
                img_original=np.reshape(image_original[x,:,:,:],[-1,image_original.shape[1],image_original.shape[2],image_original.shape[3]])
                image_averaged= avg(img_original,k=ka)
                image_gaus= do_gaus_conv(img_original,kg)
            
             
                # Clip the perturbation to make sure images fit in uint8
                clipped_v = np.clip(undo_image_avg(img_original[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(img_original[0,:,:,:]), 0, 255)
            
                
                image_perturbed = img_original + clipped_v[None, :, :, :]
                image_averaged_pert=avg(image_perturbed.astype('float32'),ka)
                image_gaus_pert=do_gaus_conv(image_perturbed.astype('float32'),size=kg)



                label_original = np.argmax(f(img_original), axis=1).flatten()
                str_label_original = labels[np.int(label_original)-1].split(',')[0]

                label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
                str_label_perturbed = labels[np.int(label_perturbed)-1].split(',')[0]

                label_average = np.argmax(f(image_averaged), axis=1).flatten()
                str_label_average_original = labels[np.int(label_average)-1].split(',')[0]

                label_gaus = np.argmax(f(image_gaus), axis=1).flatten()
                str_label_gaus_original = labels[np.int(label_gaus)-1].split(',')[0]

                label_perturbed_gaus = np.argmax(f(image_gaus_pert), axis=1).flatten()
                str_label_gaus_pert = labels[np.int(label_perturbed_gaus)-1].split(',')[0]

                label_perturbed_average = np.argmax(f(image_averaged_pert), axis=1).flatten()
                str_label_average_pert = labels[np.int(label_perturbed_average)-1].split(',')[0]

                list_out.append([names_images[x],folder,str_label_original,percent_out(img_original),str_label_perturbed,percent_out(image_perturbed),
                    str_label_average_original,percent_out(image_averaged),str_label_average_pert,percent_out(image_averaged_pert),
                    str_label_gaus_original,percent_out(image_gaus),str_label_gaus_pert,percent_out(image_gaus_pert)])


            # construct a dataframe
            import pandas as pd
            output=pd.DataFrame(list_out)
            
            output.columns=['image_name','ground truth','model prediction',' %  accuracy', 'original pert name','%  original pert name ', 'averaged name',' averaged %', 'averaged perturbation name',
            '%  of name in avg pert','gaussian name','%  gauss name','gaussian pert name','%  gaussian pert ']
            av=(output['averaged name']==folder).sum()/image_original.shape[0]*100
            gv=(output['gaussian name']==folder).sum()/image_original.shape[0]*100
            av_pert=(output['averaged perturbation name']==folder).sum()/image_original.shape[0]*100
            gv_pert=(output['gaussian pert name']==folder).sum()/image_original.shape[0]*100
            model_prediction=(output['model prediction']==folder).sum()/image_original.shape[0]*100
            pert_prediction=(output['original pert name']==folder).sum()/image_original.shape[0]*100

            
            #print('averaged {}'.format((output['averaged perturbation name']==folder).sum()))
            #print('gaussian {}'.format((output['gaussian pert name']==folder).sum()))

            k+=1
            final_list.append([folder,model_prediction,pert_prediction,av,gv,av_pert,gv_pert])
            
            inds1=os.path.join('perturbation csv files','{}.csv'.format(folder))
            output.to_csv(inds1,index=False)
            
            
            print(' >> done for class {} and saved the csv'.format(folder))
            print(' >> {} classes done and {} classes remaining'.format(k,len(dirs)-k))
        final_list=pd.DataFrame(final_list)
        final_list.columns=['ground truth','model accuracy','perturbation accuracy','averaged accuracy','gaussian accuracy','average perturbation accuracy','gaussian perturbation accuracy']

        inds1=os.path.join('perturbation csv files','final_result.csv')
        final_list.to_csv(inds1,index=False)

        print('final_result csv saved')

        # Show original and perturbed images
        '''
            plt.figure(figsize=(30,30))
            plt.subplot(3, 2, 1)
            plt.imshow(undo_image_avg(img_original[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
            plt.title('original image {}--% {}'.format(str_label_original,percent_out(img_original)))

            plt.subplot(3, 2, 2)
            plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
            plt.title('perturbed image {}--% {}'.format(str_label_perturbed,percent_out(image_perturbed)))

            plt.subplot(3, 2, 3)
            plt.imshow(undo_image_avg(image_averaged[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
            plt.title('averaged image {}--% {}'.format(str_label_average_original,percent_out(image_averaged)))

            plt.subplot(3, 2, 4)
            plt.imshow(undo_image_avg(image_averaged_pert[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
            plt.title('averaged perturbed {}--% {}'.format(str_label_perturbed,percent_out(image_averaged_pert)))

            plt.subplot(3, 2, 5)
            plt.imshow(undo_image_avg(image_gaus[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
            plt.title('gaussian image {}--% {}'.format(str_label_gaus_original,percent_out(image_gaus)))

            plt.subplot(3, 2, 6)
            plt.imshow(undo_image_avg(image_gaus_pert[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
            plt.title('gaussian perturbed {}--% {}'.format(str_label_gaus_pert,percent_out(image_gaus_pert)))

            plt.show()
        '''
