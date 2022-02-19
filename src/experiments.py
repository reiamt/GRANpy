from main import run_main
import numpy as np

#datasets = ['DREAM5_ecoli_expression_data']
datasets = ['gasch_GSE102475','gasch_GSE102475']
#ground_truths = ['DREAM5_yeast_ground_truth']
ground_truths = ['yeast_chipunion_KDUnion_intersect','yeast_chipunion_KDUnion_intersect']
epochs = [5,5]
leap_path = 'logs/leap/MAC_symmetric_gasch_example.csv'
#leap_path = 'logs/leap/MAC_symmetric_dream5_ecoli.csv'
runs = range(2)

granpy, pearson, random, leap = ([] for i in range(4))
for dataset, ground_truth, epoch in zip(datasets, ground_truths, epochs):
    for run in runs:
        granpy_scores, pearson_scores, random_scores, leap_scores = run_main(dataset, ground_truth, epoch, leap_path)
        granpy.append(granpy_scores)
        pearson.append(pearson_scores)
        random.append(random_scores)
        leap.append(leap_scores)

scores = [granpy, pearson, random, leap]
for score in scores:
    print(score)
    acc, ap, roc, f1 = zip(*score)
    for idx, dataset in enumerate(datasets):
        start = idx*(len(score)/len(datasets))
        end = (idx+1)*(len(score)/len(datasets))
        print(start)
        print(end)
        print(acc)
        print(acc[start:end])
        print('mean acc ' + str(np.mean(acc[start:end])) + ' with a sd of ' + str(np.std(acc[start:end])))
        print('mean ap ' + str(np.mean(ap[start:end])) + ' with a sd of ' + str(np.std(ap[start:end])))
        print('mean roc ' + str(np.mean(roc[start:end])) + ' with a sd of ' + str(np.std(roc[start:end])))
        print('mean f1 ' + str(np.mean(f1[start:end])) + ' with sd a of ' + str(np.std(f1[start:end])))


if len(granpy) != 0:
    print(pearson)
    print(granpy)
    #acc, ap, roc = zip(*pearson)
    #print('unzipped pearson acc for 2 runs')
    #print(acc)
    #print('average acc of the 2 runs')
    #print(np.mean(acc))
    #print(ap)
    #print(np.mean(granpy_scores))
