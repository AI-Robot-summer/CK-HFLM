import numpy as np
from Core.Tree import ClassTree

"""
Hierarchical: Hie
Classification: Cla
Evaluation: Eva
Measure: Mea
pre: predict

References
[1] Aris Kosmopoulos, Ioannis Partalas, Eric Gaussier, Georgios Paliouras, Ion Androutsopoulos, 
    Evaluation measures for hierarchical classification: a unified view and novel approaches, 
    Data Mining Knowledge Discovery, (2015) 29:820–865.
[2] O. Dekel, J. Keshet, and Y. Singer, Large __margin hierarchical classification, 
    International Conference on Machine Learning, 2004, pp. 27–35.
"""


class HieClaEvaMea(object):
    def __init__(self, class_tree: ClassTree):
        super(HieClaEvaMea, self).__init__()
        self.class_tree = class_tree
        self.class_tree_depth = self.class_tree.GetDepth()

    def Hie_Precision_Recall_F1_one_sample(self, true_y, pre_y):
        """
        Page 883 of Reference [1],
        :param true_y: the ground truth class label, belongs to {0, 1, ..., class_num-1}
        :param pre_y: the predicted class label, belongs to {0, 1, ..., class_num-1}
        :return:
        """
        precision = 1.0
        recall = 1.0
        if true_y != pre_y:
            aug_true_y = self.class_tree.GetAncestors(base_class_id=true_y)
            aug_pre_y = self.class_tree.GetAncestors(base_class_id=pre_y)
            intersection = np.intersect1d(aug_true_y, aug_pre_y)  # intersection.shape[0] >= 1
            precision = intersection.shape[0] / aug_pre_y.shape[0]
            recall = intersection.shape[0] / aug_true_y.shape[0]

        F1 = 2 * precision * recall / (precision + recall)
        return {'hie_precision': precision, 'hie_recall': recall, 'hie_F1': F1}

    def TreeInducedError_one_sample(self, true_y, pre_y):
        """
        d(u, v) is defined to be the number of edges along the (unique) path from node u to node v in Tree
        above formula (1) of Reference [2],
        :param true_y:
        :param pre_y:
        :return:
        """
        if true_y == pre_y:
            return 0.0
        else:
            return self.class_tree.GetPathLength(base_class_id1=true_y, base_class_id2=pre_y)

    def Hie_Precision_Recall_F1(self, true_y_arr: np.array, pre_y_arr: np.array):
        """

        :param true_y_arr: shape=[n_sam]
        :param pre_y_arr: shape=[n_sam]
        :return:
        """
        n_sam = true_y_arr.shape[0]
        HP_arr = -np.inf * np.ones(shape=[n_sam])
        HR_arr = -np.inf * np.ones(shape=[n_sam])
        HF1_arr = -np.inf * np.ones(shape=[n_sam])
        for i in range(n_sam):
            res = self.Hie_Precision_Recall_F1_one_sample(true_y=true_y_arr[i], pre_y=pre_y_arr[i])
            HP_arr[i] = res['hie_precision']
            HR_arr[i] = res['hie_recall']
            HF1_arr[i] = res['hie_F1']

        return {'hie_precision': np.mean(HP_arr), 'hie_recall': np.mean(HR_arr), 'hie_F1': np.mean(HF1_arr)}

    def TreeInducedError(self, true_y_arr: np.array, pre_y_arr: np.array):
        """

        :param true_y_arr: shape=[n_sam]
        :param pre_y_arr: shape=[n_sam]
        :return:
        """
        n_sam = true_y_arr.shape[0]
        err_arr = -np.inf * np.ones(shape=[n_sam])
        for i in range(n_sam):
            err_arr[i] = self.TreeInducedError_one_sample(true_y=true_y_arr[i], pre_y=pre_y_arr[i])

        return np.mean(err_arr)

    def TreeInducedError_Normalized(self, true_y_arr: np.array, pre_y_arr: np.array):
        """

        :param true_y_arr: shape=[n_sam]
        :param pre_y_arr: shape=[n_sam]
        :return:
        """
        n_sam = true_y_arr.shape[0]
        err_arr = -np.inf * np.ones(shape=[n_sam])
        for i in range(n_sam):
            err_arr[i] = self.TreeInducedError_one_sample(true_y=true_y_arr[i], pre_y=pre_y_arr[i])

        fm = 2.0 * self.class_tree_depth
        err_arr = err_arr / fm
        return np.mean(err_arr)
