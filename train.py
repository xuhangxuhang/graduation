import tensorflow as tf
import os
from os.path import join, exists

import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.layers import Input
from keras.backend.tensorflow_backend import set_session

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
    classes = int(kwargs['classes'])
    do_validate_flag = bool(int(kwargs['do_val_flag']))
    
    
    
    msg1 = 'use validation'
    msg2 = 'drop valitaion'
    if do_validate_flag: print(msg1)
    else: print(msg2)
    
    '''这个do_validation_flag是用来应对CASIA-FASD这种不带验证集的情况的
        如果有验证集，我们在训练的每个epoch之后在验证集中得出EER和ACC等指标，
            再在每经过10epoch训练后再测试集中得出指标（当然这个指标只是看是否在测试集中overfitting，不会用于模型调参）
        如果没有验证集，每轮训练后直接的错测试集的指标，这个指标只用于观察，不用于模型选型和调试参数等'''
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
    
    train_tfrecord = glob('{}/frames-{}/tfrcd/frames-{}-{}-interval-{}*.tfrecord'.format(base_dir,frame_nb,frame_nb,'train',interval))[0]
    if not do_validate_flag:
        test_tfrecord = glob('{}/frames-{}/tfrcd/frames-{}-{}-interval-{}*.tfrecord'.format(base_dir,frame_nb,frame_nb,'test',interval))[0]
    else:
        devel_tfrecord = glob('{}/frames-{}/tfrcd/frames-{}-{}-interval-{}*.tfrecord'.format(base_dir,frame_nb,frame_nb,'devel',interval))[0]
        test_tfrecord = glob('{}/frames-{}/tfrcd/frames-{}-{}-interval-{}*.tfrecord'.format(base_dir,frame_nb,frame_nb,'test',interval))[0]
        
    savedir = '{}/results/frames-{}/depth-{}'.format(base_dir,frame_nb,model_depth)
    if not os.path.exists(savedir): os.makedirs(savedir)
    
    label_csv = glob('{}/frames-{}/*.csv'.format(base_dir,frame_nb))[0]
    label_df = pd.read_csv(label_csv)
    
    
    #Callbacks
    callback_csv_logger = CSVLogger(savedir+'/model_depth_{}_training.log'.format(model_depth))
    callback_ckp_by_epoch = ModelCheckpoint(savedir+'/{epoch:03d}-{devel_loss:.5f}.hdf5',
                                       verbose=1,
                                       monitor='devel_loss',
                                       save_best_only=False,
                                       period=20,# save weights after fixed epochs
                                       mode='min')
    callback_ckp_by_perform = ModelCheckpoint(savedir+'/{epoch:03d}-{devel_loss:.5f}.hdf5',
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
    
    callback_lr_decay = LearningRateScheduler(lr_decay,verbose=0)
    
    
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
            
            metrics_str = ' '
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

        def on_epoch_end(self, epoch, logs={}):
            metrics_names = ['acc','loss','eer','auc']
            if int(epoch)%10!=0:
                self.verbose=0
                results = np.asarray((np.inf,np.inf,np.inf,np.inf))
                
            else:    
                self.test_model.set_weights(self.model.get_weights())
                y_pred = self.test_model.predict(None, None, steps=int(self.num_steps),verbose=self.verbose)
                acc = get_accuracy(self.label,y_pred)
                pred_loss = get_loss(self.label,y_pred)
                equal_error_rate, _ = get_eer(self.label,y_pred)
                auc = get_auc(self.label,y_pred)
                results = [acc,pred_loss,equal_error_rate,auc]
                
            metrics_str = ' '
            for result, name in zip(results, metrics_names):
                metric_name = self.metrics_prefix + '_' + name
                logs[metric_name] = result
                if self.verbose > 0:
                    metrics_str = metric_name + ': ' + str(result) + ' ' +metrics_str
            if self.verbose > 0:
                print(metrics_str)
    
    if not exists(savedir): os.makedirs(savedir)
    
    loss_str = 'categorical_crossentropy' if classes==2 else 'binary_crossentropy'
    metrics = {'predict':['acc']}
    loss = {'predict':loss_str}
    
    
    one_hot = True if classes ==2 else False
    
    train_data, train_label = decode_tfrecord(filenames=train_tfrecord,batch_size=batch_size,one_hot=one_hot)
    train_model = Arch(model_input=Input(tensor=train_data),block_nb=model_depth,classes=classes)
    train_target_tensor = {'predict':train_label}
    train_model.compile(optimizer=Adam(lr=1e-3),loss=loss,metrics=metrics,target_tensors=train_target_tensor)
    train_data = get_tfrecord_sample_nb(train_tfrecord)    
    train_steps_per_epoch = train_data//batch_size
    train_steps_per_epoch = 10
    
    test_data, test_label = decode_tfrecord(filenames=test_tfrecord,batch_size=batch_size,one_hot=one_hot)
    test_model  = Arch(model_input=Input(tensor=test_data),block_nb=model_depth,classes=classes)    
    test_target_tensor = {'predict':test_label}
    test_model.compile(optimizer=Adam(lr=1e-3),loss=loss,metrics=metrics,target_tensors=test_target_tensor)
    test_data = get_tfrecord_sample_nb(test_tfrecord)
    test_steps_per_epoch = test_data//batch_size
    test_steps_per_epoch = 10
    
    if do_validate_flag:
        devel_data, devel_label = decode_tfrecord(filenames=devel_tfrecord,batch_size=batch_size,one_hot=one_hot)
        devel_model = Arch(model_input=Input(tensor=devel_data),block_nb=model_depth,classes=classes)    
        devel_data = get_tfrecord_sample_nb(devel_tfrecord)    
        devel_steps_per_epoch = devel_data//batch_size  
        devel_steps_per_epoch = 10
        devel_target_tensor = {'predict':devel_label}
        devel_model.compile(optimizer=Adam(lr=1e-3),loss=loss,metrics=metrics,target_tensors=devel_target_tensor)
        callback_devel = EvaluateInputTensor(model=devel_model,steps=devel_steps_per_epoch,label=_pd_shuffle(label_df,'devel'))
        callback_test = TestInputTensor(model=test_model,steps=test_steps_per_epoch,label=_pd_shuffle(label_df,'test'))
        callbacks = [callback_devel,callback_test,callback_csv_logger,callback_ckp_by_epoch,callback_ckp_by_perform]
    else:
        callback_test = EvaluateInputTensor(model=test_model,steps=test_steps_per_epoch,label=_pd_shuffle(label_df,'test'))
        callbacks = [callback_test,callback_csv_logger,callback_ckp_by_epoch,callback_ckp_by_perform]
    
    
        
    print('start training...')
    history = train_model.fit(steps_per_epoch=train_steps_per_epoch,epochs=epoch,verbose=1,callbacks=callbacks)

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
