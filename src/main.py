import config
import datetime
import copy

from utils import write_log, set_random_seed
from train import DashFusion_train, DashFusion_test

if __name__ == '__main__':
    # follow below performance
    dataset = config.dataset
    load_metric = config.PARAM.downStream.load_metric
    check_list = config.PARAM.downStream.check_list
    metric = config.PARAM.downStream.metric
    # select which model to save
    check = config.PARAM.downStream.check
    if config.dataset == 'mosi':
        result_path = config.MOSI.result_path
    elif config.dataset == 'mosei':
        result_path = config.MOSEI.result_path
    elif config.dataset == 'sims':
        result_path = config.SIMS.result_path
    
    seed = config.seed
    result_M = {}
    for s in seed:
        config.seed = s

        set_random_seed(s)
        DashFusion_train(dataset=dataset, check=check, config=config)
        result_M[s] = DashFusion_test(dataset=dataset, check_list=check_list, config=config)

    result_path = result_path + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '.csv'
    with open(result_path, 'w') as file:
        for index, result in enumerate([
            result_M,
        ]):
            file.write('Method%s\n' % index)
            if dataset=='sims':
                file.write('seed,acc_2,F1,acc_3,acc_5,MAE,Corr\n')
                for s in seed:
                    log = '%s,%s,%s,%s,%s,%s,%s\n' % (s,
                                                    result[s][metric]['Mult_acc_2'],
                                                    result[s][metric]['F1_score'],
                                                    result[s][metric]['Mult_acc_3'],
                                                    result[s][metric]['Mult_acc_5'],
                                                    result[s][metric]['MAE'],
                                                    result[s][metric]['Corr'])
                    file.write(log)
            else:
                file.write('seed,Has0_acc_2,Has0_F1_score,Non0_acc_2,Non0_F1_score,Mult_acc_5,Mult_acc_7,MAE,Corr\n')
                for s in seed:
                    log = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (s,
                                                            result[s][metric]['Has0_acc_2'],
                                                            result[s][metric]['Has0_F1_score'],
                                                            result[s][metric]['Non0_acc_2'],
                                                            result[s][metric]['Non0_F1_score'],
                                                            result[s][metric]['Mult_acc_5'],
                                                            result[s][metric]['Mult_acc_7'],
                                                            result[s][metric]['MAE'],
                                                            result[s][metric]['Corr'])
                    file.write(log)
