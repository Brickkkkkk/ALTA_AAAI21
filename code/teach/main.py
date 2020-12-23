import matplotlib.pyplot as plt
import numpy as np
import os
import offline_teachers as teach
import datasets as ds
import utils as ut
import data_output as op
import hypothesis as hp

def teaching():
    plt.close('all')
    dataset_root = '../../data/'
    datasets = ['blobs_2_class', '2d_outlier', 'blobs_3_class', '3blobs',
                'iris', 'breast_cancer', 'wine',
                'oct', 'butterflies_crop', 'chinese_chars', 'chinese_chars_crowd', 'butterflies_crop_monarch',
        'chinese_chars_binary', 'woodpecker']
    dataset_name = datasets[11]

    experiment_id = 4
    num_teaching_itrs = 10
    num_random_test_ims = 20
    num_init_hyps = 200
    density_sigma = 1.0
    interp_sigma = 1.0
    dist_sigma = 0.001 #beta in AAAI21 paper
    alpha = 0.5
    phi = 3  # eta in AAAI21 paper
    beta = 1  # gamma in AAAI21 paper
    image_scale = 2.0
    hyp_type = 'rand'  # rand, cluster, cluster_rand, sparse
    dataset_dir = dataset_root + dataset_name + '/'
    url_root = ''  # set this to the location of the images on the web

    save_ops = False
    add_bias = True
    remove_mean = True
    do_pca = False
    pca_dims = 2


    op_dir = 'output/' + str(experiment_id) + '/'
    if save_ops:
        print('saving output to', op_dir)
        if not os.path.isdir(op_dir):
            os.makedirs(op_dir)

    # load data
    dataset_train, dataset_test = ds.load_datasets(dataset_name, dataset_dir, do_pca, pca_dims, add_bias, remove_mean, density_sigma, interp_sigma, dist_sigma)
    if len(np.unique(dataset_train['Y'])) > 2:
        one_v_all = True  # multi class
    else:
        one_v_all = False # binary


    # generate set of hypotheses
    hyps, prior_h = hp.generate_hyps(dataset_train, alpha, num_init_hyps, hyp_type, one_v_all)
    print(len(hyps), hyp_type, 'hypotheses\n')

    # remove examples that are inconsistent with best hypothesis
    if one_v_all:
        err_hyp = hp.compute_hyps_error_one_vs_all(hyps, dataset_train['X'], dataset_train['Y'], alpha)
    else:
        err_hyp = hp.compute_hyps_error(hyps, dataset_train['X'], dataset_train['Y'], alpha)
    dataset_train = ds.remove_exs(dataset_train, hyps, err_hyp, alpha, 'train', one_v_all)

    # re compute hypothesis errors - after removing inconsistent examples
    if one_v_all:
        err_hyp = hp.compute_hyps_error_one_vs_all(hyps, dataset_train['X'], dataset_train['Y'], alpha)
        err_hyp_test = hp.compute_hyps_error_one_vs_all(hyps, dataset_test['X'], dataset_test['Y'], alpha)
    else:
        err_hyp = hp.compute_hyps_error(hyps, dataset_train['X'], dataset_train['Y'], alpha)
        err_hyp_test = hp.compute_hyps_error(hyps, dataset_test['X'], dataset_test['Y'], alpha)

    # compute the likelihood for each datapoint according to each hypothesis
    if one_v_all:
        likelihood  = ut.compute_likelihood_one_vs_all(hyps, dataset_train['X'], dataset_train['Y'], alpha)
    else:
        likelihood = ut.compute_likelihood(hyps, dataset_train['X'], dataset_train['Y'], alpha)

    # teachers
    teachers = {}
    if one_v_all:
        teachers['rand_1vall'] = teach.RandomImageTeacherOneVsAll(dataset_train, alpha, prior_h)
        teachers['strict_1vall'] = teach.StrictTeacherOneVsAll(dataset_train, alpha, prior_h)
        teachers['explain_1vall'] = teach.ExplainTeacherOneVsAll(dataset_train, alpha, prior_h)
    else:
        teachers['RANDOM'] = teach.RandomImageTeacher(dataset_train, alpha, prior_h)
        teachers['STRICT'] = teach.StrictTeacher(dataset_train, alpha, prior_h)
        teachers['EXPLAIN'] = teach.ExplainTeacher(dataset_train, alpha, prior_h)
        teachers['ALTA'] = teach.ALTATeacher(dataset_train, alpha, prior_h, phi, beta)

    # run teaching
    for alg_name in list(teachers.keys()):
        print(alg_name)
        teachers[alg_name].run_teaching(num_teaching_itrs, dataset_train, likelihood, hyps, err_hyp, err_hyp_test)

    return teachers

    # plot in 2D
    # _linestyle = ['-','--','-.',':']
    # _marker = ['+','x','*','.']
    #
    # fig_id = 0
    # if (dataset_train['X'].shape[1] <= 3):
    #     for alg_name in list(teachers.keys()):
    #         if one_v_all:
    #             ut.plot_2D_data(dataset_train['X'], dataset_train['Y'], alpha, hyps, teachers[alg_name].teaching_exs, teachers[alg_name].posterior(), alg_name, fig_id, one_v_all, np.argmin(err_hyp))
    #         else:
    #             ut.plot_2D_data_hyper(dataset_train['X'], dataset_train['Y'], alpha, hyps, teachers[alg_name].teaching_exs, teachers[alg_name].posterior(), alg_name, fig_id, one_v_all, np.argmin(err_hyp))
    #         fig_id += 1
    #
    # plt.figure(fig_id)
    # plt.title('learners expected error - train')
    # for alg_name in list(teachers.keys()):
    #     exp_err = teachers[alg_name].exp_err
    #     plt.plot(np.arange(len(exp_err))+1, exp_err, label=alg_name)
    # plt.legend()
    # if save_ops:
    #     plt.savefig(op_dir + 'eer.pdf')
    #
    # plt.figure(fig_id+1)
    # plt.title('learners expected error - test')
    # i=0
    # for alg_name in list(teachers.keys()):
    #     exp_err_test = teachers[alg_name].exp_err_test
    #     plt.plot(np.arange(len(exp_err_test))+1, exp_err_test, linestyle=_linestyle[i], marker=_marker[i], label=alg_name)
    #     i += 1
    # plt.legend()
    # if save_ops:
    #     plt.savefig(op_dir + 'eer_test.pdf')
    #
    # plt.figure(fig_id+2)
    # plt.title('example difficulty')
    # for alg_name in list(teachers.keys()):
    #     difficulty = teachers[alg_name].difficulty
    #     plt.plot(np.arange(len(difficulty))+1, difficulty, label=alg_name)
    # plt.legend()
    # plt.show()
    #
    # # save strategy files
    # if not save_ops:
    #     print('\nnot saving outputs')
    # else:
    #     print('\nsaving outputs')
    #     for alg_name in list(teachers.keys()):
    #         op.save_teaching_sequence(teachers[alg_name], alg_name, op_dir + alg_name + '.strat')
    #
    #     op.save_teaching_images(dataset_train, dataset_test, op_dir + 'teaching_images.json', url_root)
    #     op.save_settings(dataset_train, dataset_test, experiment_id, num_random_test_ims, image_scale, op_dir + 'settings.json')
    #     np.savez(op_dir + 'params.npz', dataset_train=dataset_train, dataset_test=dataset_test, hyps=hyps, teachers=teachers)

iteration = 30

exp_err_test = {}
alg_names = ['RANDOM', 'STRICT', 'EXPLAIN', 'ALTA']
for x in alg_names:
    exp_err_test[x] = np.zeros(10)
for i in range(iteration):
    print(i)
    teachers = teaching()
    i = 0
    for alg_name in list(teachers.keys()):
        exp_err_test[alg_name] += teachers[alg_name].exp_err_test

_linestyle = ['-','--','-.',':']
_marker = ['v','x','s','.']


plt.figure(figsize=(4,3))
plt.title('learners expected error - test')
i=0
for alg_name in alg_names:
    exp_err_test[alg_name] = exp_err_test[alg_name]/iteration
    plt.plot(np.arange(len(exp_err_test[alg_name]))+1, exp_err_test[alg_name], linestyle=_linestyle[i], marker=_marker[i], label=alg_name, linewidth=1.5)
    i += 1
plt.legend()
op_dir = 'output/' + '0' + '/'
plt.savefig(op_dir + 'eer_test.pdf')
#np.savez(op_dir + 'notime.npz', exp_err_test=exp_err_test['ALTA'])
plt.show()
