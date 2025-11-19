import torch
from tqdm import tqdm
import transformers as trans
import datetime
import numpy as np
import matplotlib.pyplot as plt

import config as default_config
from model.dashfusion import DashFusion
from dataloader.MOSI import MOSIDataloader
from dataloader.MOSEI import MOSEIDataloader
from dataloader.SIMS import SIMSDataloader
from utils import write_log, check_and_save, Metrics

def update_matrix(data, model, dataset, config=default_config):
    with torch.no_grad():
        model.eval()
        if dataset == 'mosi':
            train_data = MOSIDataloader('train', batch_size=128, use_similarity=False, simi_return_mono=False, 
                                    shuffle=False,use_sampler=False)
        elif dataset == 'mosei':
            train_data = MOSEIDataloader('train', batch_size=128, use_similarity=False, simi_return_mono=False,
                                     shuffle=False, use_sampler=False)
        elif dataset == 'sims':
            train_data = SIMSDataloader('train', batch_size=128, use_similarity=False, simi_return_mono=False,
                                     shuffle=False, use_sampler=False)
        
        device = config.DEVICE
        print('Collecting New embeddings')
        T, V, A = [], [], []
        bar = tqdm(train_data, disable=True)
        for index, sample in enumerate(bar):
            _T, _V, _A = model(sample, None, return_loss=False)
            T.append(_T.detach())
            V.append(_V.detach())
            A.append(_A.detach())
        T = torch.cat(T, dim=0).to(torch.device('cpu')).squeeze()
        V = torch.cat(V, dim=0).to(torch.device('cpu')).squeeze()
        A = torch.cat(A, dim=0).to(torch.device('cpu')).squeeze()
        print('Updating Similarity Matrix')
        data.dataset.update_matrix(T, V, A)
        model.train()
    return


def DashFusion_train(dataset=None, check=None, config=default_config):
    print('---------------DsahFusion_EXP_---------------')
    if check is None:
        check = {'Loss': 10000, 'MAE': 100}
    else:
        check = check.copy()
    metrics = Metrics()
    train_bool = [True, True, True, True, True]

    model = DashFusion(config=config)

    model.set_train(train_bool)

    device = config.DEVICE
    dataset = config.dataset
    batch_size = config.PARAM.Train.batch_size
    lr = config.PARAM.Train.lr
    total_epoch = config.PARAM.Train.epoch
    decay = config.PARAM.Train.decay
    num_warm_up = config.PARAM.Train.num_warm_up

    if dataset == 'mosi':
        train_data = MOSIDataloader('train', batch_size=batch_size, use_similarity=True, simi_return_mono=False, use_sampler=False)
        log_path = config.MOSI.log_path + "experiment." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '.log'
    elif dataset == 'mosei':
        train_data = MOSEIDataloader('train', batch_size=batch_size, use_similarity=True, simi_return_mono=False, use_sampler=False)
        log_path = config.MOSEI.log_path + "experiment." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '.log'
    elif dataset == 'sims':
        train_data = SIMSDataloader('train', batch_size=batch_size, use_similarity=True, simi_return_mono=False, use_sampler=False)
        log_path = config.SIMS.log_path + "experiment." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '.log'
    
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr, amsgrad=False, )
    scheduler = trans.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        num_warm_up * (len(train_data))),num_training_steps=total_epoch * len(train_data),)
    model.to(device)


    loss = 0
    all_loss = 0
    const_loss = 0
    save_start_epoch = 1
    for epoch in range(1, total_epoch + 1):
        if epoch % 2 == 1:
            update_matrix(train_data, model, dataset, config)
        model.train()
        bar = tqdm(train_data, disable=False)
        
        for index, sample1, in enumerate(bar):
            try:
                bar.set_description("Epoch:%d|All_loss:%s|Loss:%s|Const_loss:%s" % (
                    epoch, all_loss.item(), loss.item(), const_loss.item()))
            except:
                bar.set_description(
                    "Epoch:%d|All_loss:%.6f|Loss:%.6f|Const_loss:%.6f" % (epoch, all_loss, loss, const_loss))

            optimizer.zero_grad()

            idx = sample1['index']
            sample2 = train_data.dataset.sample(idx)

            pred, all_loss, loss, const_loss = model(sample1, sample2, return_loss=True)
            
            all_loss.backward()
            optimizer.step()
            scheduler.step()

        print("EVAL valid")
        result, result_loss = eval(model, dataset, metrics, 'valid', device, config)
        if dataset=='sims':
            log1 = 'Valid_Metric Epoch:%d\n\tacc_2:%s\n\tF1_score:%s\n\tacc_3"%s\n\t' \
                'acc_5:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '---------------------------------------' % (epoch, 
                result['Mult_acc_2'], result['F1_score'], result['Mult_acc_3'], 
                result['Mult_acc_5'], result['MAE'], result['Corr'], result_loss)     

        else :
            log1 = 'Valid_Metric Epoch%s:\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
                'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '-----------------------------------------' % (epoch,
                result['Has0_acc_2'],result['Has0_F1_score'],result['Non0_acc_2'], result['Non0_F1_score'],
                result['Mult_acc_5'],result['Mult_acc_7'], result['MAE'],result['Corr'],result_loss)   
        
        print(log1)
        write_log(log1, path=log_path)

        if epoch > save_start_epoch:
            check = check_and_save(model, dataset, result, check, save_model=True)


def eval(model, dataset=None, metrics=None, eval_data=None, device=None, config=default_config):
    batch_size = config.PARAM.Train.batch_size
    with torch.no_grad():
        model.eval()
        if device is None: device = config.DEVICE
        plt_flag=False
        if eval_data is None:
            plt_flag=True
            if dataset == 'mosi':
                eval_data = MOSIDataloader('test', shuffle=False, num_workers=0, batch_size=batch_size)
            elif dataset == 'mosei':
                eval_data = MOSEIDataloader('test', shuffle=False, num_workers=0, batch_size=batch_size)
            elif dataset == 'sims':
                eval_data = SIMSDataloader('test', shuffle=False, num_workers=0, batch_size=batch_size)
        else:
            if dataset == 'mosi':
                eval_data = MOSIDataloader(eval_data, shuffle=False, num_workers=0, batch_size=batch_size)
            elif dataset == 'mosei':
                eval_data = MOSEIDataloader(eval_data, shuffle=False, num_workers=0, batch_size=batch_size)
            elif dataset == 'sims':
                eval_data = SIMSDataloader(eval_data, shuffle=False, num_workers=0, batch_size=batch_size)
        if metrics is None: metrics = Metrics()
        pred = []
        truth = []
        loss = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            label = sample['regression_labels'].clone().detach().to(device).float()
            _pred, _all_loss, _loss, _, = model(sample, None, return_loss=True)
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.item() * config.PARAM.Train.batch_size
        
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        if dataset =='sims':
            eval_results = metrics.eval_sims_regression(truth, pred)
        else:
            eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / len(eval_data)
        model.train()

    return eval_results, loss / len(eval_data)


def DashFusion_test(dataset=None, check_list=None, config=default_config):
    if check_list is None: check_list = ['Has0_F1_score', 'MAE']
    if not isinstance(check_list, list): check_list = [check_list]
    seed = config.seed

    model = DashFusion(config=config)

    device = config.DEVICE
    model.to(device)
    check = {}
    result = None
    print('Evaluating model')
    for metric in check_list:
        print('Result for best ' + metric)
        model.load_model(name='best_'+metric, dataset=dataset)
        result, loss = eval(model=model, dataset=dataset, device=device, config=config)
        check[metric] = {}
        for key in result.keys():
            check[metric][key] = result[key]

        if dataset=='sims':
            log = 'Test_Metric\n\tacc_2:%s\n\tF1_score:%s\n\tacc_3:%s\n\t' \
                'acc_5:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '------------------------------------------' % (result['Mult_acc_2'], result['F1_score'],
                                                                result['Mult_acc_3'], result['Mult_acc_5'],
                                                                result['MAE'], result['Corr'], loss)                                                                                                   

        else:
            log = 'Test_Metric\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
                'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
                '------------------------------------------' % (result['Has0_acc_2'], result['Has0_F1_score'],
                                                                result['Non0_acc_2'], result['Non0_F1_score'],
                                                                result['Mult_acc_5'], result['Mult_acc_7'],
                                                                result['MAE'], result['Corr'], loss)
                                                                
        print(log)

    return check
