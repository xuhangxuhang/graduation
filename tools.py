import numpy as np
from os.path import basename
from sklearn.metrics import accuracy_score, log_loss, roc_curve, roc_auc_score

def get_tfrecord_sample_nb(tfrecord):
    return int(basename(tfrecord).split('.')[0].split('-')[-1])

def get_eer(y_true,y_pred):
    '''https://github.com/maciej3031/whale-recognition-challenge/blob/30a9ef774ef344916bbc8dce4229fa005d346a64/utils/performance_utils.py#L310'''
    fpr, tpr, thrd = roc_curve(y_true,y_pred,pos_label=1)
    frr = 1 - tpr
    index_eer = np.argmin(abs(fpr - frr))
    eer = (fpr[index_eer] + frr[index_eer])/2
    eer_thrd = thrd[index_eer]
    return eer, eer_thrd

def get_hter(y_true,y_pred,eer_thd=None):
    '''please input eer_thd, else eer of test set will be calculated'''
    fpr, tpr, thrd = roc_curve(y_true,y_pred,pos_label=1)
    frr = 1 - tpr
    if eer_thd is None:
        index = np.argmin(abs(fpr - frr))
        hter = min((fpr[index]+frr[index])/2)
    else:
        hter_thrd = thrd[np.argmin(thrd-eer_thd)]
        hter = (fpr[hter_thrd]+frr[hter_thrd])/2
    return hter

def apcer_bpcer_acer(y_true,y_pred,PAI_type_1_idxs,PAI_type_2_idxs):
    '''
    在计算APCER，BPCER和ACER时需要指出属于纸张攻击和视频攻击的indexs
    因为我们这里attack只有两种：print_attack and replay_attack(video_attack)
    故只有两种PAI_types
    
    return apcer, bpcer and acer
    '''

    bona = 0  # 活脸标签为0
    attack = 1  # 非活脸标签为1
    
    '''BPCER'''
    # step-1 找出y_true中的真脸对应的index
    true_bona_idxs = np.where(y_true==bona)[0]
    # step-2 找出y_pred中属于step-1的index的预测标签
    pred_values_belong_to_bona = y_pred[true_bona_idxs]
    # step-3 判断这些值和真实值的差别
    nb_pred_bona_correctly_classified = len(pred_values_belong_to_bona)
    # step-4 APCER
    bpcer = nb_pred_bona_correctly_classified/len(true_bona_idxs)
    
    '''APCERs'''
    # 预测attack_type_1的结果
    pred_attack_type_1 = y_pred[PAI_type_1_idxs]
    # 预测后attack_type_1中correctly预测出attack的样本数量
    nb_pred_attack_type_1_correctly_classified = len(np.where(pred_attack_type_1==attack)[0])
    apcer_candi_type_1 = nb_pred_attack_type_1_correctly_classified/len(PAI_type_1_idxs)
    
    # 预测attack_type_2的结果
    pred_attack_type_2 = y_pred[PAI_type_2_idxs]
    # 预测后attack_type_2中correctly预测出attack的样本数量
    nb_pred_attack_type_2_correctly_classified = len(np.where(pred_attack_type_2==attack)[0])
    apcer_candi_type_2 = nb_pred_attack_type_2_correctly_classified/len(PAI_type_2_idxs)
    
    # APCER是纸张攻击和视频攻击中值较高的那个
    apcer = max(apcer_candi_type_1,apcer_candi_type_2)
    
    acer = (apcer+bpcer)/2
    
    
    return apcer, bpcer, acer
    

def get_accuracy(y_true,y_pred):
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_true,y_pred)
    return accuracy

def get_loss(y_true,y_pred):
    return log_loss(y_true,y_pred,eps=1e-7)

def get_auc(y_true,y_pred):
    return roc_auc_score(y_true,y_pred)
