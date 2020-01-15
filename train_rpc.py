import torch
import utils
import os
import model_utils
import pickle
import read_utils_rpc
import torch.optim as optim
import numpy as np
import sklearn.metrics.pairwise as skp
import eval_ijba
from random import randint

import time
from tqdm import tqdm
import datetime
data = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

K = 1 # [optional] to eliminate too small templates while testing
margin = 2.0  # follow the original paper
feat_type = 'resnet34' # casianet or resnet34
set_size = 8  # size of image set
batch_size = 64  # number of subjects per batch
set_per_sub = 3  # number of image set per subject within a batch
pooling_type = 'NAN'
if feat_type == 'casianet':
    feat_dim = 320
else:
    feat_dim = 512

max_iter = 51000
test_iter = 50

save_dir = './data/IJBA/model/model_{}_s{}_{}'.format(feat_type, set_size, pooling_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logger = utils.config_log(save_dir, 'train_s{}b{}'.format(set_size, batch_size))
meta_dir = './data/IJBA/{}'.format(feat_type)
logger_result = open(os.path.join(save_dir, 'eval_result.txt'), 'w')
NAN_split_tars = np.zeros((10, 4), dtype=np.float32)
AVE_split_tars = np.zeros((10, 4), dtype=np.float32)
for idx_split in range(1, 11):
    # lst_sub_faces = pickle.load(open(os.path.join(meta_dir, 'train_subject_{}.bin'.format(idx_split)), 'rb'))
    label_qry = pickle.load(open('label_qry20200114145205rpc.p', 'rb'))
    label_ref = pickle.load(open('label_ref20200115092149rpc.p', 'rb'))
    feature_qry = torch.load('feature_qry20200114145205rpc.p')
    feature_ref = torch.load('feature_ref20200115092149rpc.p')

    lst_sub_faces = torch.cat((feature_qry, feature_ref), 0)
    label = label_qry + label_ref
    set_size = len(label_ref)//len(set(label_ref))
    count_qry = len(label_qry)
    label_ref_sub = label_ref[0::15]

    dataset = read_utils_rpc.Dataset(feature_ref, batch_size, set_size, label)
    '''
    a = [i for i, x in enumerate(feature_ref) if x == ""]
    
    from collections import Counter
    print(Counter(label_ref))
    label_ref.count('196')

    '''

    lst_sub_faces = set(label_ref)

    # dataset = read_utils.Dataset(lst_sub_faces, batch_size, set_size)
    net = model_utils.Pooling_Classifier(feat_dim=feat_dim, num_classes=200, pooling_type=pooling_type)
    criterion = torch.nn.CrossEntropyLoss()
    contrastive_criterion = model_utils.Online_Contrastive_Loss(num_classes=200)
    # optimizer_nn = optim.Adam(net.parameters(), weight_decay=5e-5)
    optimizer_nn = optim.RMSprop(net.parameters(), lr=0.001, weight_decay=1e-5)

    lst_tar = []
    for iter in range(max_iter):
        iter += 1
        batch, lst_label, lst_len = dataset.next_pair_batch(set_per_sub)
        '''
        np.random.seed(55)
        ref_len = len(label_ref)
        ref_beg = randint(0,int(ref_len/8-1))
        if int(ref_len/8-1) -ref_beg < batch_size/8:
            label_ref_sub = label_ref[8*ref_beg:] + label_ref[:batch_size-(ref_len-8*ref_beg)]
            feature_ref_sub =torch.cat((feature_ref[8*ref_beg:], feature_ref[:batch_size-(ref_len-8*ref_beg)]), 0)
        else:
            label_ref_sub = label_ref[ref_beg*8:8*ref_beg+batch_size]
            feature_ref_sub = feature_ref[ref_beg*8:8*ref_beg+batch_size]
        counter ={}
        for w in label_ref_sub:
            counter[w] = counter.get(w, 0) + 1

        batch = feature_ref_sub
        lst_label = [int(x) for x in label_ref_sub]
        lst_len = list(counter.values())
        '''

        # batch, lst_label, lst_len = dataset.next_pair_batch(set_per_sub)
        batch = np.vstack(batch)
        # batch = skp.normalize(batch)
        lst_len = np.array(lst_len)
        lst_label = np.array([int(y) for y in [label_ref[8*x] for x in lst_label]])

        batch = torch.from_numpy(batch).float()
        lst_len = torch.from_numpy(lst_len)
        targets = torch.from_numpy(lst_label).long()

        feats, logits = net(batch, lst_len)
        # loss_sf = criterion(logits, targets)
        loss_contrastive = contrastive_criterion(feats, targets)
        loss = loss_contrastive  # + 0.0*loss_sf
        _, predicted = torch.max(logits.data, 1)
        # accuracy = (targets.data == predicted).float().mean()
        optimizer_nn.zero_grad()
        loss.backward()
        if iter > 2:
            optimizer_nn.step()
        if iter % test_iter == 0:
            with torch.no_grad():
                feats, _ = net(feature_ref, torch.IntTensor([set_size]*len(set(label_ref))))
                feature_distance1 = torch.mm(feature_qry, feats.t())
                top3_val, top3_pos = torch.topk(feature_distance1, 3)
                top3_pos = top3_pos.tolist()
                val_corrects1 = 0
                val_corrects2 = 0
                val_corrects3 = 0
                val_rsift_corrects = 0

                print('Processing features and Refinement')
                count_bar_refine = tqdm(total=count_qry)
                for ind_qry, ind_ref in enumerate(top3_pos):
                    count_bar_refine.update(1)
                    top3_label = [label_ref_sub[i] for i in ind_ref]
                    if label_qry[ind_qry] == top3_label[0]:
                        val_corrects1 = val_corrects1 + 1
                    occurrences = top3_label.count(label_qry[ind_qry])
                    if top3_label.count(label_qry[ind_qry]):
                        val_corrects3 = val_corrects3 + 1
                    if top3_label[:2].count(label_qry[ind_qry]):
                        val_corrects2 = val_corrects2 + 1
                count_bar_refine.close()

                val_acc1 = val_corrects1 / count_qry
                val_acc2 = val_corrects2 / count_qry
                val_acc3 = val_corrects3 / count_qry
                val_rsift_corrects = val_rsift_corrects / count_qry
                accuracy = 'val_acc1:{acc1}, val_acc2:{acc2}, val_acc3:{acc3}, val_rsift:{rsift}'. \
                    format(acc1=val_acc1, acc2=val_acc2, acc3=val_acc3, rsift=val_rsift_corrects)
                print(accuracy)

                #save_path = os.path.join('weights_%.4f_%.4f_%.4f.pth' % (
                #val_acc1, val_acc2, val_acc3))
                # torch.cuda.synchronize()
                #torch.save(net.state_dict(), save_path)
                #print('saved model to %s' % (save_path), flush=True)
                # torch.cuda.empty_cache()

                torch.save(feats, 'feature_ref{a}{b}.p'.format(a=data, b='rpc'))
                # torch.save(features_qry[:count_qry], 'feature_qry{a}{b}.p'.format(a=data, b=args.dataset))
                # torch.load('feature_ref')
                import pickle

                pickle.dump(label_ref_sub, open('label_ref{a}{b}.p'.format(a=data, b='rpc'), 'wb'))
                # pickle.dump(labels_qry, open('label_qry{a}{b}.p'.format(a=data, b=args.dataset), 'wb'))

                now = time.strftime("%X, %x")
                with open('accuracy.txt', "a") as log_file:
                    log_file.write(
                        '{time}_data_{data}:{accuracy}\n'.format(time=now, data='rpc', accuracy=accuracy))

                print('iter:{iter}  loss:{a}, accuracy:{b}'.format(iter=iter,a=loss, b = accuracy))


        '''
        if iter > 2:
            optimizer_nn.step()

        if iter % test_iter == 0 or iter == 1:
            lst_test_pair = pickle.load(
                open(os.path.join(meta_dir, 'lst_test_pair_{}.bin'.format(idx_split)), 'rb'))

            # cur_pair = 0
            # c_sim = np.zeros(shape=[len(lst_test_pair)], dtype=np.float32)
            # actual_issame = np.zeros(shape=[len(lst_test_pair)], dtype=np.bool)
            c_sim = []
            actual_issame= []
            for pair in lst_test_pair:
                vfea_a = pair[0]
                vfea_b = pair[1]
                if vfea_a.shape[0] < K or vfea_b.shape[0] < K:
                    continue

                batch_a = torch.from_numpy(vfea_a).float()
                lst_len = torch.from_numpy(np.array([vfea_a.shape[0]]))
                feats, logits = net(batch_a, lst_len)
                mfeat_a = feats.detach().numpy()

                batch_b = torch.from_numpy(vfea_b).float()
                lst_len = torch.from_numpy(np.array([vfea_b.shape[0]]))
                feats, logits = net(batch_b, lst_len)
                mfeat_b = feats.detach().numpy()

                nfeat_a = eval_ijba.norm_l2(mfeat_a)
                nfeat_b = eval_ijba.norm_l2(mfeat_b)
                cos_d = np.dot(nfeat_b, np.transpose(nfeat_a))
                # c_sim[cur_pair] = cos_d
                issame = pair[2]
                # actual_issame[cur_pair] = issame
                c_sim.append(cos_d)
                actual_issame.append(issame)
                # cur_pair += 1
            # end of pair
            c_sim = np.array(c_sim)
            actual_issame = np.array(actual_issame)
            fars, tars, thrs, FA, TA, acc = eval_ijba.cal_far(c_sim, actual_issame)
            lst_tar.append(np.expand_dims(np.array(tars), axis=0))
            np_lst_tar = np.vstack(lst_tar)
            idx_max = np.argmax(np_lst_tar[:, 1])
            # print('# split {} pair {}'.format(idx_split, len(c_sim)))
            logger.info(
                'iter {} loss {} loss_c {} accuracy {}'.format(iter, loss.data, loss_contrastive.data, accuracy.data))
            logger.info('split {} cur tar {}'.format(idx_split, tars))
            logger.info('split {} max tar {}'.format(idx_split, np_lst_tar[idx_max, :]))
            logger.info('\n')
            if iter == 1:
                logger_result.write('split {} init tar {} \n'.format(idx_split, np_lst_tar))
                AVE_split_tars[idx_split - 1, :] = np_lst_tar

        # end of testing
    # end of training
    np_lst_tar = np.vstack(lst_tar)
    idx_max = np.argmax(np_lst_tar[:, 1])
    logger_result.write('split {} max tar {} \n'.format(idx_split, np_lst_tar[idx_max, :]))
    NAN_split_tars[idx_split - 1, :] = np_lst_tar[idx_max, :]
    logger_result.write('AVE Mean Result {} \n'.format(np.mean(AVE_split_tars[0:idx_split, :], axis=0)))
    logger_result.write('AVE STD Result {} \n'.format(np.std(AVE_split_tars[0:idx_split, :], axis=0)))
    logger_result.write('NAN Mean Result {} \n'.format(np.mean(NAN_split_tars[0:idx_split, :], axis=0)))
    logger_result.write('NAN STD Result {} \n'.format(np.std(NAN_split_tars[0:idx_split, :], axis=0)))

    logger.info('AVE Mean Result {}'.format(np.mean(AVE_split_tars[0:idx_split, :], axis=0)))
    logger.info('AVE STD Result {}'.format(np.std(AVE_split_tars[0:idx_split, :], axis=0)))
    logger.info('NAN Mean Result {}'.format(np.mean(NAN_split_tars[0:idx_split, :], axis=0)))
    logger.info('NAN STD Result {}'.format(np.std(NAN_split_tars[0:idx_split, :], axis=0)))
    logger_result.flush()
# end of splits
utils.save(NAN_split_tars, os.path.join(save_dir, 'NAN_result.bin'))
utils.save(AVE_split_tars, os.path.join(save_dir, 'AVE_result.bin'))
logger_result.close()
'''
