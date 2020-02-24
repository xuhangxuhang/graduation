import tensorflow as tf
import os
from os.path import join, exists

import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.layers import Input
from keras.callbacks import TensorBoard

from tfrecord_server import decode_tfrecord,_pd_shuffle
from model import Arch
from keras.callbacks import LearningRateScheduler

from glob import glob
import numpy as np
import pandas as pd
import math
import argparse
import configparser
from tools import get_tfrecord_sample_nb

from tools import get_eer, get_accuracy, get_loss, get_auc

def train(**kwargs):

    interval = kwargs['interval']
    base_dir = kwargs['base_dir']
    model_depth = int(kwargs['model_depth'])
    batch_size = int(kwargs['batch_size'])
    epoch = int(kwargs['epoch'])
    gpuId = str(kwargs['gpuid'])
    frame_nb = kwargs['frame_nb']
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
    
    train_tfrecord = glob('{}/frames-{}/tfrcd/frames-{}-{}-interval-{}*.tfrecord'.format(base_dir,frame_nb,frame_nb,'train',interval))[0]
    devel_tfrecord = glob('{}/frames-{}/tfrcd/frames-{}-{}-interval-{}*.tfrecord'.format(base_dir,frame_nb,frame_nb,'devel',interval))[0]
    test_tfrecord = glob('{}/frames-{}/tfrcd/frames-{}-{}-interval-{}*.tfrecord'.format(base_dir,frame_nb,frame_nb,'test',interval))[0]
    
    
    savedir = '{}/results/frames-{}/depth-{}'.format(base_dir,frame_nb,model_depth)
    if not os.path.exists(savedir): os.makedirs(savedir)
    
    label_csv = glob('{}/frames-{}/*.csv'.format(base_dir,frame_nb))[0]
    label_df = pd.read_csv(label_csv)
    
    
    #Callbacks
    csv_logger = CSVLogger(savedir+'/model_depth_{}_training.log'.format(model_depth))
    ckp_by_epoch = ModelCheckpoint(savedir+'/{epoch:03d}-{devel_loss:.5f}.hdf5',
                                       verbose=1,
                                       monitor='devel_loss',
                                       save_best_only=False,
                                       period=20,# save weights after fixed epochs
                                       mode='min')
    ckp_by_perform = ModelCheckpoint(savedir+'/{epoch:03d}-{devel_loss:.5f}.hdf5',
                                       verbose=1,
                                       monitor='devel_loss',
                                       save_best_only=True,
                                       mode='min')
    def lr_decay(epoch):
        ''' lr drop start from 100th epoch, drop rate is 0.5, drop frequency is 20
            if necessary change initial lrate in this function, make sure that this 
            initial learning rate equals to it in Adam Optimizer'''
        initial_lr = 1e-3
        drop_start_epoch = 30
        drop = 0.5
        epochs_drop = 30
        lr_minim = 1e-4
        lrate = initial_lr*math.pow(drop, math.floor((1+epoch-drop_start_epoch)/epochs_drop)) if epoch>drop_start_epoch else initial_lr
        lrate = max(lrate,lr_minim)
        return lrate
    
    Learning_rate_decay = LearningRateScheduler(lr_decay,verbose=0)
    
    
    class EvaluateInputTensor(Callback):
        def __init__(self,model,steps,label,metrics_prefix='devel',verbose=1):
            super(EvaluateInputTensor, self).__init__()
            self.devel_model = model
            self.verbose = verbose
            self.num_steps = steps
            self.label = label
            self.metrics_prefix = metrics_prefix

        def on_epoch_end(self, epoch, logs={}):
            self.devel_model.set_weights(self.model.get_weights())
            y_pred = self.devel_model.predict(None,None,steps=int(self.num_steps),verbose=self.verbose)
            acc = get_accuracy(self.label,y_pred)
            pred_loss = get_loss(self.label,y_pred)
            equal_error_rate, _ = get_eer(self.label,y_pred)
            auc = get_auc(self.label,y_pred)
            
            metrics_names = ['acc','loss','eer','auc']
            results = [acc,pred_loss,equal_error_rate,auc]
            
            metrics_str = '\n'
            for result, name in zip(results, metrics_names):
                metric_name = self.metrics_prefix + '_' + name
                logs[metric_name] = result
                if self.verbose > 0:
                    metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '
            if self.verbose > 0:
                print(metrics_str) 
                
    class TestInputTensor(Callback):
        def __init__(self,model,steps,label,metrics_prefix='test',verbose=1):
            super(TestInputTensor, self).__init__()
            self.test_model = model
            self.verbose = verbose
            self.num_steps = steps
            self.label = label
            self.metrics_prefix = metrics_prefix
            
        metrics_names = ['acc','loss','eer','auc']

        def on_epoch_end(self, epoch, logs={}):
            if int(epoch)%10!=0:
                self.verbose=0
                y_pred = np.asarray((np.inf,np.inf,np.inf,np.inf))
            else:    
                self.test_model.set_weights(self.model.get_weights())
                y_pred = self.test_model.evaluate(None, None, steps=int(self.num_steps),verbose=self.verbose)
            metrics_str = '\n'
            for result, name in zip(y_pred, metrics_names):
                metric_name = self.metrics_prefix + '_' + name
                logs[metric_name] = result
                if self.verbose > 0:
                    metrics_str = metric_name + ': ' + str(result) + ' ' +metrics_str
            if self.verbose > 0:
                print(metrics_str)
    
    if not exists(savedir): os.makedirs(savedir)
    
    # load data
    train_data, train_label = decode_tfrecord(filenames=train_tfrecord,batch_size=batch_size,one_hot=False)
    devel_data, devel_label = decode_tfrecord(filenames=devel_tfrecord,batch_size=batch_size,one_hot=False)
    test_data, test_label = decode_tfrecord(filenames=test_tfrecord,batch_size=batch_size,one_hot=False)

    metrics = {'predict':['acc']}
    
    print(train_label)
    
    # gene models
    train_model = Arch(model_input=Input(tensor=train_data),block_nb=model_depth,classes=1)
    devel_model = Arch(model_input=Input(tensor=devel_data),block_nb=model_depth,classes=1)
    test_model  = Arch(model_input=Input(tensor=test_data),block_nb=model_depth,classes=1)
    
    
#     embedding_data_func = decode_tfrecord(filenames=test_tfrecord,batch_size=100,repeat_count=1,one_hot=True)
#     with tf.Session() as sess:
#         embedding_data = sess.run(embedding_data_func)
#     print('successfully unpacked embedding data...')
    
#     tb = TensorBoard(log_dir=savedir+'/model_depth_{}_tensorboard_logs'.format(model_depth),
#                 write_images=False,histogram_freq=1,write_graph=False)
#     embedding_layer_names = [train_model.layers[-6].name,train_model.layers[-1].name]
#     tb_dir = savedir+'/model_depth_{}_tensorboard_logs'.format(model_depth)
#     if not exists(tb_dir): os.makedirs(tb_dir)
    
#     with open(join(tb_dir, 'metadata.tsv'), 'w') as f:
#         np.savetxt(f, embedding_data[1])
#     tb = TensorBoard(log_dir=tb_dir,
#                      write_images=True,
#                      write_graph=True,
#                      write_grads=True,
#                      embeddings_freq=2,
#                      embeddings_layer_names=embedding_layer_names,
#                      embeddings_metadata='metadata.tsv',
#                      embeddings_data=embedding_data[0])
    
    loss = {'predict':'binary_crossentropy'}
    train_target_tensor = {'predict':train_label}
    train_model.compile(optimizer=Adam(lr=1e-3),loss=loss,metrics=metrics,target_tensors=train_target_tensor)
    

    devel_target_tensor = {'predict':devel_label}
    devel_model.compile(optimizer=Adam(lr=1e-3),loss=loss,metrics=metrics,target_tensors=devel_target_tensor)
    
    test_target_tensor = {'predict':test_label}
    test_model.compile(optimizer=Adam(lr=1e-3),loss=loss,metrics=metrics,target_tensors=test_target_tensor)

    train_data = get_tfrecord_sample_nb(train_tfrecord)
    train_steps_per_epoch = train_data//batch_size
    
    devel_data = get_tfrecord_sample_nb(devel_tfrecord)
    devel_steps_per_epoch = devel_data//batch_size
    
    test_data = get_tfrecord_sample_nb(test_tfrecord)
    test_steps_per_epoch = test_data//batch_size
    
    
    devel_callback = EvaluateInputTensor(model=devel_model,steps=devel_steps_per_epoch,label=_pd_shuffle(label_df,'devel'))
    test_callback = TestInputTensor(model=test_model,steps=test_steps_per_epoch,label=_pd_shuffle(label_df,'test'))
        
    print('start training...')
    history = train_model.fit(steps_per_epoch=train_steps_per_epoch,
                              epochs=epoch,
                              verbose=1,
                              callbacks=[Learning_rate_decay,devel_callback
                                         ,test_callback,csv_logger,
                                         ckp_by_epoch,
                                         ckp_by_perform])

    train_model.save_weights(savedir+'/final_model.hdf5')
    pd.to_pickle(history,savedir+'/history.pkl')
    
    return history


if __name__ == '__main__':
    ## 管理参数，创建实例
    parser = argparse.ArgumentParser(description='configuration file')
    #  '-conf': 配置文件
    parser.add_argument('-conf',help='configure file')
    arg = parser.parse_args()
    config_filename = arg.conf
    
    #生成config对象
    config = configparser.ConfigParser()
    #用config对象读取配置文件  
    config.read(config_filename)
    
    input_dict = {}
    
    print(config.sections())
    section = config[config.sections()[0]]

    for key in section:
        input_dict[key] = section[key]
    
    hisory = train(**input_dict)