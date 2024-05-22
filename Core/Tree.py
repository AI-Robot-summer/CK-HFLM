import numpy as np
import pickle
from copy import copy
from typing.io import IO
import nltk
from nltk.corpus import wordnet as wn
from pyecharts import options as opts
from pyecharts.charts import Tree

from Core.DataSet import ClassSpaceInformation
from Core.QuotientSpace import Partition
from Core.FuzzyQuotientSpace import HierarchicalPartition
from Core.Kit import List

"""
Cla: Class
BFS: Breadth First Search
"""


class TreeNode(object):
    """
    构造函数的参数的默认值不能等于[]，例如：base_class_id_list=[]，否则会出错
    """

    def __init__(self, synset_=None, base_class_id_list=None):
        """

        :param synset_: wn.Synset or string
        """
        super(TreeNode, self).__init__()

        # wordnet.synset
        self.synset_ = synset_

        #  list of class_id,  class_id belongs to {0, 1, 2, ..., num_class-1}
        if base_class_id_list is None:
            self.base_class_id_list = []
        else:
            self.base_class_id_list = base_class_id_list

        # TreeNode
        self.parent = None

        # list of TreeNode
        self.children = []

        # int
        self.BSF_id = -999

    def Show(self):
        print(str(self))

    def Save2File(self, f_writer: IO):
        f_writer.write(str(self) + '\n')
        f_writer.flush()

    def AppendChild(self, child_node):
        # print('AppendChild()@' + str(self.synset_.name()))
        self.children.append(child_node)

    def DeleteInvalidChild(self):
        """
        A node is Invalid
        if ond only if
        it doest not include any base class
        :return:
        """
        old_children = self.children
        self.children = []
        for old_ch in old_children:
            if old_ch.IncludeBaseClass():
                self.children.append(old_ch)

    def SetParent(self, parent_node):
        self.parent = parent_node

    def AppendBaseClassID(self, class_id):
        self.base_class_id_list.append(class_id)

    def AppendBaseClassIDRecursion(self, class_id):
        node = self
        while node is not None:
            if not node.base_class_id_list.__contains__(class_id):
                node.base_class_id_list.append(class_id)
                node = node.parent
            else:
                node = None

    def IsLeaf(self):
        return len(self.children) == 0

    def IsRoot(self):
        return self.parent is None

    def IncludeBaseClass(self):
        return len(self.base_class_id_list) > 0

    def GetClassWord(self):
        name = self.synset_.name()  # word.pos.nn, e.g. dog.n.01
        return name.split('.')[0]

    def GetChildContains(self, base_class_id):
        """

        :param base_class_id: belongs to {0, 1, 2, ..., num_class-1}
        :return:
        """
        for ch_node in self.children:
            if ch_node.base_class_id_list.__contains__(base_class_id):
                return ch_node
        return None

    def SynsetName(self):
        if type(self.synset_) is str:
            return self.synset_
        else:
            return self.synset_.name()

    def BaseClassStr(self):
        if self.base_class_id_list.__len__() == 0:
            return '{}'
        arr = np.array(self.base_class_id_list)
        arr = np.sort(arr)
        rep_str = '{' + str(arr[0])
        for i in range(1, arr.shape[0]):
            rep_str += ', '
            rep_str += str(arr[i])
        rep_str += '}'
        return rep_str

        # if self.base_class_id_list.__len__() == 1:
        #     return '{' + str(self.base_class_id_list[0]) + '}'
        # elif self.base_class_id_list.__len__() == 2:
        #     return '{' + str(self.base_class_id_list[0]) + ', ' + \
        #                  str(self.base_class_id_list[1]) + '}'
        # elif self.base_class_id_list.__len__() == 3:
        #     return '{' + str(self.base_class_id_list[0]) + ', ' + \
        #                  str(self.base_class_id_list[1]) + ', ' + \
        #                  str(self.base_class_id_list[2]) + '}'
        # elif self.base_class_id_list.__len__() > 3:
        #     return '{' + str(self.base_class_id_list[0]) + ', ' + \
        #                  str(self.base_class_id_list[1]) + ', ' + \
        #                  str(self.base_class_id_list[2]) + ', ..., ' + \
        #                  str(self.base_class_id_list[-1]) + '}'

    def ToPyechartsDict_WordNet(self):
        self_dict = {"name": str(self.BSF_id) + '@' + str(self.synset_)}
        if not self.IsLeaf():
            ch_list = list()
            for ch_node in self.children:
                ch_list.append(ch_node.ToPyechartsDict_WordNet())
            self_dict["children"] = ch_list
        else:
            if type(self.synset_) is str:
                name = self.synset_
            else:
                name = str(self.synset_)
            self_dict = {"name": str(self.BSF_id) + '@' + name}
        return self_dict

    def ToPyechartsDict(self):
        self_dict = {"name": str(self.BSF_id)}
        if not self.IsLeaf():
            ch_list = list()
            for ch_node in self.children:
                ch_list.append(ch_node.ToPyechartsDict())
            self_dict["children"] = ch_list
        else:
            if type(self.synset_) is str:
                name = self.synset_
            else:
                name = self.synset_.name()
                name = name.split('.')[0]
            self_dict = {"name": str(self.BSF_id) + '@' + name}
        return self_dict

    def ToPyechartsDict_CIFAR10(self):
        if type(self.synset_) is str:
            name = self.synset_
        else:
            name = self.synset_.name()
            name = name.split('.')[0]
        self_dict = {"name": str(self.BSF_id) + '@' + name}
        if not self.IsLeaf():
            ch_list = list()
            for ch_node in self.children:
                ch_list.append(ch_node.ToPyechartsDict_CIFAR10())
            self_dict["children"] = ch_list
        return self_dict


class ClassTree(object):
    counter = -1

    def __init__(self):
        self.class_inf = None

        # TreeNode
        self.root_node = None

        # list of TreeNode
        self.leaf_node_list = []

        # list of TreeNode
        self.base_class_node_list = []

    @staticmethod
    def __GetBaseClassID(node: TreeNode, class_word_pos_nn_id_map):
        if class_word_pos_nn_id_map is None:
            return None
        else:
            class_word_pos_nn = node.synset_.name()
            class_id = class_word_pos_nn_id_map.get(class_word_pos_nn)
            if class_id is not None:
                del class_word_pos_nn_id_map[class_word_pos_nn]
            return class_id

    @staticmethod
    def __GetBaseClassID2_list(parent_node: TreeNode, cla_id_parent_synset_map):
        """
        追加自定义的上/下位关系，一个上位词对应多个下位词
        :param parent_node:
        :param cla_id_parent_synset_map:
        :return:
        """
        cla_id_sublist = []
        if type(parent_node.synset_) is not nltk.corpus.reader.wordnet.Synset:
            return cla_id_sublist
        cla_id_list = cla_id_parent_synset_map.keys()
        for cla_id in cla_id_list:
            parent_synset = cla_id_parent_synset_map[cla_id]
            if parent_node.synset_.__eq__(parent_synset):
                cla_id_sublist.append(cla_id)
        for cla_id in cla_id_sublist:
            del cla_id_parent_synset_map[cla_id]
        return cla_id_sublist

    def __WordNet2Tree(self):
        # print("Wordnet {}".format(wn.get_version()))
        class_word_pos_nn_id_map = None
        if self.class_inf is not None:
            class_word_pos_nn_id_map = copy(self.class_inf.class_word_pos_nn_id_map)

        self.root_node = TreeNode(synset_=wn.synset('entity.n.01'))
        queue = List()
        queue.InQueue(self.root_node)
        # i = 1
        while not queue.IsEmpty():
            node = queue.OutQueue()
            class_id = ClassTree.__GetBaseClassID(node, class_word_pos_nn_id_map)
            if class_id is not None:
                node.AppendBaseClassIDRecursion(class_id=class_id)
                self.base_class_node_list.append(node)

            # print(str(i) + ' ' + str(node.synset_.name()))
            # i += 1

            hyponym_list = node.synset_.hyponyms()
            if len(hyponym_list) > 0:
                for item in hyponym_list:
                    child_node = TreeNode(synset_=item)
                    child_node.SetParent(parent_node=node)
                    node.AppendChild(child_node=child_node)
                    # print(' ' + str(self.root_node.synset_) + '# children= ' + str(len(self.root_node.children)))
                    queue.InQueue(child_node)
            else:
                self.leaf_node_list.append(node)
        return class_word_pos_nn_id_map

    def __PruneTree(self):
        queue = List()
        queue.InQueue(self.root_node)
        while not queue.IsEmpty():
            node = queue.OutQueue()
            node.DeleteInvalidChild()
            queue.InQueue_list(node.children)

        self.leaf_node_list = []
        queue = List()
        queue.InQueue(self.root_node)
        while not queue.IsEmpty():
            node = queue.OutQueue()
            if node.IsLeaf():
                self.leaf_node_list.append(node)
                # print(node.base_class_id_list)
            else:
                queue.InQueue_list(node.children)

    def __AddLeafNode_not_in_WordNet(self, base_class_id_list):
        cla_id_parent_synset_map = {}
        for cla_id in base_class_id_list:
            other = self.class_inf.other_list[cla_id]
            # e.g. cartman is hyponym of wn.synset('cartoon.n.01')
            word_pos_nn = other.split('\'')[1]
            parent_synset = wn.synset(word_pos_nn)
            cla_id_parent_synset_map[cla_id] = parent_synset

        queue = List()
        queue.InQueue(self.root_node)
        while not queue.IsEmpty():
            node = queue.OutQueue()
            cla_id_list = ClassTree.__GetBaseClassID2_list(parent_node=node,
                                                           cla_id_parent_synset_map=cla_id_parent_synset_map)
            for cla_id in cla_id_list:
                node_str = self.class_inf.class_word_list[cla_id] + ' is hyponym of ' + str(node.synset_)
                child = TreeNode(synset_=node_str)
                child.AppendBaseClassID(class_id=cla_id)
                node.AppendChild(child_node=child)
                child.SetParent(parent_node=node)
                node.AppendBaseClassIDRecursion(class_id=cla_id)
                self.base_class_node_list.append(child)
                self.leaf_node_list.append(child)
            if cla_id_parent_synset_map.__len__() == 0:
                break

            if not node.IsLeaf():
                queue.InQueue_list(ele_list=node.children)

    def UpdateBSF_id(self):
        ClassTree.counter = 1
        queue = List()
        queue.InQueue(ele=self.root_node)
        while not queue.IsEmpty():
            node = queue.OutQueue()
            node.BSF_id = ClassTree.counter
            ClassTree.counter += 1
            if not node.IsLeaf():
                queue.InQueue_list(ele_list=node.children)

    def GetNodeNum(self):
        if self.root_node.BSF_id < 0:
            self.UpdateBSF_id()
        node_num = - 1
        for leaf in self.leaf_node_list:
            if leaf.BSF_id > node_num:
                node_num = leaf.BSF_id

        return node_num

    def GetLeafNum(self):
        return len(self.leaf_node_list)

    def GetDepth(self):
        depth = -1
        for leaf in self.leaf_node_list:
            leaf_depth = self.GetNodeDepth(node=leaf)
            if leaf_depth > depth:
                depth = leaf_depth
        return depth

    def GetNodeDepth(self, node: TreeNode):
        depth = 1
        while node.parent is not None:
            depth += 1
            node = node.parent

        return depth

    def GetNode(self, base_class_id):
        for i in range(len(self.base_class_node_list)):
            node = self.base_class_node_list[i]
            base_class_id_list = node.base_class_id_list
            if len(base_class_id_list) == 1 and base_class_id_list[0] == base_class_id:
                return node

        return None

    def GetAncestors(self, base_class_id) -> np.array:
        """

        :param base_class_id: belongs to
        :return: the BSF_id of Ancestors of node corresponding to base_class_id
        """

        node = self.GetNode(base_class_id=base_class_id)
        BSF_id_arr = []
        while node is not None:
            BSF_id_arr.append(node.BSF_id)
            node = node.parent

        return np.array(BSF_id_arr)

    def GetLowestCommonAncestor(self, base_class_id1, base_class_id2):
        """

        :param base_class_id1: belongs to {0, 1, 2, ..., num_class-1}
        :param base_class_id2: belongs to {0, 1, 2, ..., num_class-1}
        :return:
        """
        node = self.GetNode(base_class_id=base_class_id1)
        while not node.base_class_id_list.__contains__(base_class_id2):
            node = node.parent
        return node

    def GetPathLength(self, base_class_id1, base_class_id2):
        """

        :param base_class_id1: belongs to {0, 1, 2, ..., num_class-1}
        :param base_class_id2: belongs to {0, 1, 2, ..., num_class-1}
        :return:
        """
        path_len = 0
        node = self.GetNode(base_class_id=base_class_id1)
        while not node.base_class_id_list.__contains__(base_class_id2):
            node = node.parent
            path_len += 1

        while node.base_class_id_list.__len__() > 1:
            node = node.GetChildContains(base_class_id=base_class_id2)
            path_len += 1

        return path_len

    def Save2File(self, file_name):
        bytes = pickle.dumps(self)
        with open(file=file_name, mode='wb') as f:
            f.write(bytes)

    def ToDictFile(self, file_name):
        """Test that a nontrivial hierarchy leaf classification behaves as expected.

           We build the following class hierarchy along with data from the handwritten digits dataset:
           From

                   <ROOT>
                  /      \
                 A        B
                / \       |  \
               1   7      C   9
                        /   \
                       3     8

           To
            dict = {
            ROOT: ["A", "B"],
            "A": ["1", "7"],
            "B": ["C", "9"],
            "C": ["3", "8"],
            }
           """
        f = open(file=file_name, mode='w')
        f.write('class_hierarchy = {\n')
        queue = List()
        queue.InQueue(self.root_node)
        while not queue.IsEmpty():
            node = queue.OutQueue()
            if not node.IsLeaf():
                if node.BSF_id > 1:
                    f.write('\t' + str(node.BSF_id) + ': ')
                else:
                    f.write('\t' + 'ROOT: ')
                f.write(' [' + str(node.children[0].BSF_id))
                queue.InQueue(node.children[0])
                num_child = node.children.__len__()
                for i in range(1, num_child):
                    f.write(', ' + str(node.children[i].BSF_id))
                    queue.InQueue(node.children[i])
                f.write('],\n')
        f.write('}\n')
        f.flush()

        f.write('class_BSf_id_map = {\n')
        for node in self.base_class_node_list:
            if node.base_class_id_list.__len__() > 1:
                print('node.base_class_id_list.__len__() > 1')
            f.write('\t' + str(node.base_class_id_list[0]) + ': ' + str(node.BSF_id) + ',\n')
        f.write('}\n')
        f.flush()

        f.write('BSf_class_id_map = {\n')
        for node in self.base_class_node_list:
            if node.base_class_id_list.__len__() > 1:
                print('node.base_class_id_list.__len__() > 1')
            f.write('\t' + str(node.BSF_id) + ': ' + str(node.base_class_id_list[0]) + ',\n')
        f.write('}\n')
        f.close()

    @staticmethod
    def LoadFromFile(file_name):
        with open(file=file_name, mode='rb') as f:
            obj = pickle.load(file=f)
        return obj.as_type(ClassTree)

    @staticmethod
    def __Layer2Partition(node_list):
        equ_cla_list = []
        for node in node_list:
            equ_cla_list.append(node.base_class_id_list)
        return Partition(equ_cla_list=equ_cla_list)

    def ToHierarchicalPartition(self):
        """

        :return:
        """
        par_list = []
        layer_node_list = [self.root_node]
        while layer_node_list is not None:
            par = ClassTree.__Layer2Partition(node_list=layer_node_list)
            # if par_list.__len__() == 15:
            #     print(par_list.__len__())
            # print(par_list.__len__())
            par.Check()
            par_list.append(par)
            finer_layer = []
            is_finer = False
            for node in layer_node_list:
                if node.IsLeaf():  # there is no child, append itself
                    finer_layer.append(node)
                else:  # there are some children, append his children
                    finer_layer += node.children
                    is_finer = True
            if is_finer:
                layer_node_list = finer_layer
            else:
                layer_node_list = None
        return HierarchicalPartition(partition_list=par_list)

    @staticmethod
    def ConstructWordNet_Tree():
        tree = ClassTree()
        tree.__WordNet2Tree()
        tree.UpdateBSF_id()
        return tree

    @staticmethod
    def ShowWordNet_Tree(tree, html_file_name):
        tree_data = [tree.root_node.ToPyechartsDict(), ]
        tree = Tree(init_opts=opts.InitOpts(width="64000px", height="4000px")).add(
            collapse_interval=100,
            series_name="tree",
            data=tree_data,
            # layout="radial",
            layout="orthogonal",
            orient="TB",
            pos_top='5%',
            pos_left='5%',
            pos_bottom='5%',
            pos_right='5%',
            initial_tree_depth=-1,
            label_opts=opts.LabelOpts(is_show=True, position='right', distance=1),
            leaves_label_opts=opts.LabelOpts(is_show=True, position='bottom', rotate=60, color='red', distance=1),
        )
        tree.set_global_opts()
        tree.render(path=html_file_name)

    @staticmethod
    def ConstructCIFAR10_Tree(class_inf: ClassSpaceInformation):
        tree = ClassTree()
        tree.class_inf = class_inf
        tree.__WordNet2Tree()
        tree.__PruneTree()
        tree.UpdateBSF_id()
        return tree

    @staticmethod
    def ShowCIFAR10_Tree(tree, html_file_name):
        tree_data = [tree.root_node.ToPyechartsDict_CIFAR10(), ]
        tree = Tree().add(
            collapse_interval=0,
            series_name="tree",
            data=tree_data,
            # layout="radial",
            layout="orthogonal",
            orient="TB",
            pos_top='5%',
            pos_left='5%',
            pos_bottom='5%',
            pos_right='5%',
            initial_tree_depth=15,
            label_opts=opts.LabelOpts(position='inside', distance=1, font_size=4),
            leaves_label_opts=opts.LabelOpts(position='inside', distance=1, color='red', font_size=6),
        )
        tree.set_global_opts()
        tree.render(path=html_file_name)

    @staticmethod
    def ConstructCIFAR100_Tree(class_inf: ClassSpaceInformation):
        tree = ClassTree()
        tree.class_inf = class_inf
        tree.__WordNet2Tree()
        print('tree.__WordNet2Tree() end')
        tree.__PruneTree()
        print('tree.__PruneTree() end')
        tree.UpdateBSF_id()
        print('tree.UpdateBSF_id() end')
        return tree

    @staticmethod
    def Show(tree, html_file_name):
        tree_data = [tree.root_node.ToPyechartsDict_WordNet(), ]
        tree = Tree().add(
            collapse_interval=100,
            series_name="tree",
            data=tree_data,
            # layout="radial",
            layout="orthogonal",
            orient="TB",
            pos_top='5%',
            pos_left='5%',
            pos_bottom='5%',
            pos_right='5%',
            initial_tree_depth=-1,
            label_opts=opts.LabelOpts(position='inside', distance=1, font_size=8),
            leaves_label_opts=opts.LabelOpts(position='inside', rotate=-90, color='red', distance=-2, font_size=12,
                                             # vertical_align='top',
                                             horizontal_align='left'
                                             ),
        )
        tree.set_global_opts()
        tree.render(path=html_file_name)

    @staticmethod
    def ConstructCIFAR100_Tree_Expert():
        """
        100 classes
        """
        c0 = TreeNode(synset_='apples', base_class_id_list=[0])
        c1 = TreeNode(synset_='aquarium fish', base_class_id_list=[1])
        c2 = TreeNode(synset_='baby', base_class_id_list=[2])
        c3 = TreeNode(synset_='bear', base_class_id_list=[3])
        c4 = TreeNode(synset_='beaver', base_class_id_list=[4])
        c5 = TreeNode(synset_='bed', base_class_id_list=[5])
        c6 = TreeNode(synset_='bee', base_class_id_list=[6])
        c7 = TreeNode(synset_='beetle', base_class_id_list=[7])
        c8 = TreeNode(synset_='bicycle', base_class_id_list=[8])
        c9 = TreeNode(synset_='bottles', base_class_id_list=[9])
        c10 = TreeNode(synset_='bowls', base_class_id_list=[10])
        c11 = TreeNode(synset_='boy', base_class_id_list=[11])
        c12 = TreeNode(synset_='bridge', base_class_id_list=[12])
        c13 = TreeNode(synset_='bus', base_class_id_list=[13])
        c14 = TreeNode(synset_='butterfly', base_class_id_list=[14])
        c15 = TreeNode(synset_='camel', base_class_id_list=[15])
        c16 = TreeNode(synset_='cans', base_class_id_list=[16])
        c17 = TreeNode(synset_='castle', base_class_id_list=[17])
        c18 = TreeNode(synset_='caterpillar', base_class_id_list=[18])
        c19 = TreeNode(synset_='cattle', base_class_id_list=[19])
        c20 = TreeNode(synset_='chair', base_class_id_list=[20])
        c21 = TreeNode(synset_='chimpanzee', base_class_id_list=[21])
        c22 = TreeNode(synset_='clock', base_class_id_list=[22])
        c23 = TreeNode(synset_='cloud', base_class_id_list=[23])
        c24 = TreeNode(synset_='cockroach', base_class_id_list=[24])
        c25 = TreeNode(synset_='couch', base_class_id_list=[25])
        c26 = TreeNode(synset_='crab', base_class_id_list=[26])
        c27 = TreeNode(synset_='crocodile', base_class_id_list=[27])
        c28 = TreeNode(synset_='cups', base_class_id_list=[28])
        c29 = TreeNode(synset_='dinosaur', base_class_id_list=[29])
        c30 = TreeNode(synset_='dolphin', base_class_id_list=[30])
        c31 = TreeNode(synset_='elephant', base_class_id_list=[31])
        c32 = TreeNode(synset_='flatfish', base_class_id_list=[32])
        c33 = TreeNode(synset_='forest', base_class_id_list=[33])
        c34 = TreeNode(synset_='fox', base_class_id_list=[34])
        c35 = TreeNode(synset_='girl', base_class_id_list=[35])
        c36 = TreeNode(synset_='hamster', base_class_id_list=[36])
        c37 = TreeNode(synset_='house', base_class_id_list=[37])
        c38 = TreeNode(synset_='kangaroo', base_class_id_list=[38])
        c39 = TreeNode(synset_='computer keyboard', base_class_id_list=[39])
        c40 = TreeNode(synset_='lamp', base_class_id_list=[40])
        c41 = TreeNode(synset_='lawn-mower', base_class_id_list=[41])
        c42 = TreeNode(synset_='leopard', base_class_id_list=[42])
        c43 = TreeNode(synset_='lion', base_class_id_list=[43])
        c44 = TreeNode(synset_='lizard', base_class_id_list=[44])
        c45 = TreeNode(synset_='lobster', base_class_id_list=[45])
        c46 = TreeNode(synset_='man', base_class_id_list=[46])
        c47 = TreeNode(synset_='maple', base_class_id_list=[47])
        c48 = TreeNode(synset_='motorcycle', base_class_id_list=[48])
        c49 = TreeNode(synset_='mountain', base_class_id_list=[49])
        c50 = TreeNode(synset_='mouse', base_class_id_list=[50])
        c51 = TreeNode(synset_='mushrooms', base_class_id_list=[51])
        c52 = TreeNode(synset_='oak', base_class_id_list=[52])
        c53 = TreeNode(synset_='oranges', base_class_id_list=[53])
        c54 = TreeNode(synset_='orchids', base_class_id_list=[54])
        c55 = TreeNode(synset_='otter', base_class_id_list=[55])
        c56 = TreeNode(synset_='palm', base_class_id_list=[56])
        c57 = TreeNode(synset_='pears', base_class_id_list=[57])
        c58 = TreeNode(synset_='pickup truck', base_class_id_list=[58])
        c59 = TreeNode(synset_='pine', base_class_id_list=[59])
        c60 = TreeNode(synset_='plain', base_class_id_list=[60])
        c61 = TreeNode(synset_='plates', base_class_id_list=[61])
        c62 = TreeNode(synset_='poppies', base_class_id_list=[62])
        c63 = TreeNode(synset_='porcupine', base_class_id_list=[63])
        c64 = TreeNode(synset_='possum', base_class_id_list=[64])
        c65 = TreeNode(synset_='rabbit', base_class_id_list=[65])
        c66 = TreeNode(synset_='raccoon', base_class_id_list=[66])
        c67 = TreeNode(synset_='ray', base_class_id_list=[67])
        c68 = TreeNode(synset_='road', base_class_id_list=[68])
        c69 = TreeNode(synset_='rocket', base_class_id_list=[69])
        c70 = TreeNode(synset_='roses', base_class_id_list=[70])
        c71 = TreeNode(synset_='sea', base_class_id_list=[71])
        c72 = TreeNode(synset_='seal', base_class_id_list=[72])
        c73 = TreeNode(synset_='shark', base_class_id_list=[73])
        c74 = TreeNode(synset_='shrew', base_class_id_list=[74])
        c75 = TreeNode(synset_='skunk', base_class_id_list=[75])
        c76 = TreeNode(synset_='skyscraper', base_class_id_list=[76])
        c77 = TreeNode(synset_='snail', base_class_id_list=[77])
        c78 = TreeNode(synset_='snake', base_class_id_list=[78])
        c79 = TreeNode(synset_='spider', base_class_id_list=[79])
        c80 = TreeNode(synset_='squirrel', base_class_id_list=[80])
        c81 = TreeNode(synset_='streetcar', base_class_id_list=[81])
        c82 = TreeNode(synset_='sunflowers', base_class_id_list=[82])
        c83 = TreeNode(synset_='sweet peppers', base_class_id_list=[83])
        c84 = TreeNode(synset_='table', base_class_id_list=[84])
        c85 = TreeNode(synset_='tank', base_class_id_list=[85])
        c86 = TreeNode(synset_='telephone', base_class_id_list=[86])
        c87 = TreeNode(synset_='television', base_class_id_list=[87])
        c88 = TreeNode(synset_='tiger', base_class_id_list=[88])
        c89 = TreeNode(synset_='tractor', base_class_id_list=[89])
        c90 = TreeNode(synset_='train', base_class_id_list=[90])
        c91 = TreeNode(synset_='trout', base_class_id_list=[91])
        c92 = TreeNode(synset_='tulips', base_class_id_list=[92])
        c93 = TreeNode(synset_='turtle', base_class_id_list=[93])
        c94 = TreeNode(synset_='wardrobe', base_class_id_list=[94])
        c95 = TreeNode(synset_='whale', base_class_id_list=[95])
        c96 = TreeNode(synset_='willow', base_class_id_list=[96])
        c97 = TreeNode(synset_='wolf', base_class_id_list=[97])
        c98 = TreeNode(synset_='woman', base_class_id_list=[98])
        c99 = TreeNode(synset_='worm', base_class_id_list=[99])

        """
        20 super classes
        """
        s0 = TreeNode(synset_='aquatic mammals')
        s0.base_class_id_list = [4, 30, 55, 72, 99]
        s0.children = [c4, c30, c55, c72, c95]
        c4.parent = s0
        c30.parent = s0
        c55.parent = s0
        c72.parent = s0
        c95.parent = s0

        s1 = TreeNode(synset_='fish')
        s1.base_class_id_list = [1, 32, 67, 73, 91]
        s1.children = [c1, c32, c67, c73, c91]
        c1.parent = s1
        c32.parent = s1
        c67.parent = s1
        c73.parent = s1
        c91.parent = s1

        s2 = TreeNode(synset_='flowers')
        s2.base_class_id_list = [54, 62, 70, 82, 92]
        s2.children = [c54, c62, c70, c82, c92]
        c54.parent = s2
        c62.parent = s2
        c70.parent = s2
        c82.parent = s2
        c92.parent = s2

        s3 = TreeNode(synset_='food containers')
        s3.base_class_id_list = [9, 10, 16, 28, 61]
        s3.children = [c9, c10, c16, c28, c61]
        c9.parent = s3
        c10.parent = s3
        c16.parent = s3
        c28.parent = s3
        c61.parent = s3

        s4 = TreeNode(synset_='fruit and vegetables')
        s4.base_class_id_list = [0, 51, 53, 57, 83]
        s4.children = [c0, c51, c53, c57, c83]
        c0.parent = s4
        c51.parent = s4
        c53.parent = s4
        c57.parent = s4
        c83.parent = s4

        s5 = TreeNode(synset_='household electrical devices')
        s5.base_class_id_list = [22, 39, 40, 86, 87]
        s5.children = [c22, c39, c40, c86, c87]
        c22.parent = s5
        c39.parent = s5
        c40.parent = s5
        c86.parent = s5
        c87.parent = s5

        s6 = TreeNode(synset_='household furniture')
        s6.base_class_id_list = [5, 20, 25, 84, 94]
        s6.children = [c5, c20, c25, c84, c94]
        c5.parent = s6
        c20.parent = s6
        c25.parent = s6
        c84.parent = s6
        c94.parent = s6

        s7 = TreeNode(synset_='insects')
        s7.base_class_id_list = [6, 7, 14, 18, 24]
        s7.children = [c6, c7, c14, c18, c24]
        c6.parent = s7
        c7.parent = s7
        c14.parent = s7
        c18.parent = s7
        c24.parent = s7

        s8 = TreeNode(synset_='large carnivores')
        s8.base_class_id_list = [3, 42, 43, 88, 97]
        s8.children = [c3, c42, c43, c88, c97]
        c3.parent = s8
        c42.parent = s8
        c43.parent = s8
        c88.parent = s8
        c97.parent = s8

        s9 = TreeNode(synset_='large man-made outdoor things')
        s9.base_class_id_list = [12, 17, 37, 68, 76]
        s9.children = [c12, c17, c37, c68, c76]
        c12.parent = s9
        c17.parent = s9
        c37.parent = s9
        c68.parent = s9
        c76.parent = s9

        s10 = TreeNode(synset_='large natural outdoor scenes')
        s10.base_class_id_list = [23, 33, 49, 60, 71]
        s10.children = [c23, c33, c49, c60, c71]
        c23.parent = s10
        c33.parent = s10
        c49.parent = s10
        c60.parent = s10
        c71.parent = s10

        s11 = TreeNode(synset_='large omnivores and herbivores')
        s11.base_class_id_list = [15, 19, 21, 31, 38]
        s11.children = [c15, c19, c21, c31, c38]
        c15.parent = s11
        c19.parent = s11
        c21.parent = s11
        c31.parent = s11
        c38.parent = s11

        s12 = TreeNode(synset_='medium-sized mammals')
        s12.base_class_id_list = [34, 63, 64, 66, 75]
        s12.children = [c34, c63, c64, c66, c75]
        c34.parent = s12
        c63.parent = s12
        c64.parent = s12
        c66.parent = s12
        c75.parent = s12

        s13 = TreeNode(synset_='non-insect invertebrates')
        s13.base_class_id_list = [26, 45, 77, 79, 99]
        s13.children = [c26, c45, c77, c79, c99]
        c26.parent = s13
        c45.parent = s13
        c77.parent = s13
        c79.parent = s13
        c99.parent = s13

        s14 = TreeNode(synset_='people')
        s14.base_class_id_list = [2, 11, 35, 46, 98]
        s14.children = [c2, c11, c35, c46, c98]
        c2.parent = s14
        c11.parent = s14
        c35.parent = s14
        c46.parent = s14
        c98.parent = s14

        s15 = TreeNode(synset_='reptiles')
        s15.base_class_id_list = [27, 29, 44, 78, 93]
        s15.children = [c27, c29, c44, c78, c93]
        c27.parent = s15
        c29.parent = s15
        c44.parent = s15
        c78.parent = s15
        c93.parent = s15

        s16 = TreeNode(synset_='small mammals')
        s16.base_class_id_list = [36, 50, 65, 74, 80]
        s16.children = [c36, c50, c65, c74, c80]
        c36.parent = s16
        c50.parent = s16
        c65.parent = s16
        c74.parent = s16
        c80.parent = s16

        s17 = TreeNode(synset_='trees')
        s17.base_class_id_list = [47, 52, 56, 59, 96]
        s17.children = [c47, c52, c56, c59, c96]
        c47.parent = s17
        c52.parent = s17
        c56.parent = s17
        c59.parent = s17
        c96.parent = s17

        s18 = TreeNode(synset_='vehicles 1')
        s18.base_class_id_list = [8, 13, 48, 58, 90]
        s18.children = [c8, c13, c48, c58, c90]
        c8.parent = s18
        c13.parent = s18
        c48.parent = s18
        c58.parent = s18
        c90.parent = s18

        s19 = TreeNode(synset_='vehicles 2')
        s19.base_class_id_list = [41, 69, 81, 85, 89]
        s19.children = [c41, c69, c81, c85, c89]

        """
        root
        """
        root = TreeNode(synset_='CIFAR100-root')
        root.base_class_id_list = list(range(100))
        root.children = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19]
        s0.parent = root
        s1.parent = root
        s2.parent = root
        s3.parent = root
        s4.parent = root
        s5.parent = root
        s6.parent = root
        s7.parent = root
        s8.parent = root
        s9.parent = root
        s10.parent = root
        s11.parent = root
        s12.parent = root
        s13.parent = root
        s14.parent = root
        s15.parent = root
        s16.parent = root
        s17.parent = root
        s18.parent = root
        s19.parent = root

        tree = ClassTree()
        tree.root_node = root
        tree.base_class_node_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
                                     c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                                     c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,
                                     c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
                                     c40, c41, c42, c43, c44, c45, c46, c47, c48, c49,
                                     c50, c51, c52, c53, c54, c55, c56, c57, c58, c59,
                                     c60, c61, c62, c63, c64, c65, c66, c67, c68, c69,
                                     c70, c71, c72, c73, c74, c75, c76, c77, c78, c79,
                                     c80, c81, c82, c83, c84, c85, c86, c87, c88, c89,
                                     c90, c91, c92, c93, c94, c95, c96, c97, c98, c99]
        tree.leaf_node_list = tree.base_class_node_list
        tree.UpdateBSF_id()
        return tree

    @staticmethod
    def ConstructCaltech256_Tree(class_inf: ClassSpaceInformation):
        tree = ClassTree()
        tree.class_inf = class_inf
        rest_word_pos_nn_id_map = tree.__WordNet2Tree()
        res_cla_id_li = list(rest_word_pos_nn_id_map.values())
        tree.__AddLeafNode_not_in_WordNet(base_class_id_list=res_cla_id_li)
        tree.__PruneTree()
        tree.UpdateBSF_id()
        return tree

    @staticmethod
    def ShowCaltech256_Tree(tree, html_file_name):
        tree_data = [tree.root_node.ToPyechartsDict(), ]
        tree = Tree(init_opts=opts.InitOpts(width="5000px", height="800px")).add(
            collapse_interval=300,
            series_name="tree",
            data=tree_data,
            # layout="radial",
            layout="orthogonal",
            orient="TB",
            pos_top='5%',
            pos_left='5%',
            pos_bottom='5%',
            pos_right='5%',
            initial_tree_depth=-1,
            label_opts=opts.LabelOpts(position='right', distance=1),
            leaves_label_opts=opts.LabelOpts(position='bottom', rotate=60, color='red', distance=1),
        )
        tree.set_global_opts()
        tree.render(path=html_file_name)

    @staticmethod
    def ConstructPASCAL_VOL_tree_Expert():
        # base class #
        c0 = TreeNode(synset_='aeroplane', base_class_id_list=[0])
        c1 = TreeNode(synset_='bird', base_class_id_list=[1])
        c2 = TreeNode(synset_='bicycle', base_class_id_list=[2])
        c3 = TreeNode(synset_='boat', base_class_id_list=[3])
        c4 = TreeNode(synset_='bottle', base_class_id_list=[4])
        c5 = TreeNode(synset_='bus', base_class_id_list=[5])
        c6 = TreeNode(synset_='car', base_class_id_list=[6])
        c7 = TreeNode(synset_='cat', base_class_id_list=[7])
        c8 = TreeNode(synset_='chair', base_class_id_list=[8])
        c9 = TreeNode(synset_='cow', base_class_id_list=[9])
        c10 = TreeNode(synset_='dining table', base_class_id_list=[10])
        c11 = TreeNode(synset_='dog', base_class_id_list=[11])
        c12 = TreeNode(synset_='horse', base_class_id_list=[12])
        c13 = TreeNode(synset_='motorbike', base_class_id_list=[13])
        c14 = TreeNode(synset_='person', base_class_id_list=[14])
        c15 = TreeNode(synset_='potted plant', base_class_id_list=[15])
        c16 = TreeNode(synset_='sheep', base_class_id_list=[16])
        c17 = TreeNode(synset_='sofa', base_class_id_list=[17])
        c18 = TreeNode(synset_='train', base_class_id_list=[18])
        c19 = TreeNode(synset_='tv/monitor', base_class_id_list=[19])

        # superclass 4th layer #
        s4_0 = TreeNode(synset_='seating')
        s4_0.base_class_id_list = [8, 17]
        s4_0.children = [c8, c17]
        c8.parent = s4_0
        c8.parent = s4_0

        # superclass 3th layer #
        s3_0 = TreeNode(synset_='2-wheeled')
        s3_0.base_class_id_list = [2, 13]
        s3_0.children = [c2, c13]
        c2.parent = s3_0
        c13.parent = s3_0

        s3_1 = TreeNode(synset_='4-wheeled')
        s3_1.base_class_id_list = [5, 6]
        s3_1.children = [c5, c6]
        c5.parent = s3_1
        c6.parent = s3_1

        s3_2 = TreeNode(synset_='domestic')
        s3_2.base_class_id_list = [7, 11]
        s3_2.children = [c7, c11]
        c7.parent = s3_2
        c11.parent = s3_2

        s3_3 = TreeNode(synset_='furniture')
        s3_3.base_class_id_list = [8, 10, 17]
        s3_3.children = [c10, s4_0]
        c10.parent = s3_3
        s4_0.parent = s3_3

        s3_4 = TreeNode(synset_='farmyard')
        s3_4.base_class_id_list = [9, 12, 16]
        s3_4.children = [c9, c12, c16]
        c9.parent = s3_4
        c12.parent = s3_4
        c16.parent = s3_4

        # superclass 2nd layer #
        s2_0 = TreeNode(synset_='animal')
        s2_0.base_class_id_list = [1, 7, 9, 11, 12, 16]
        s2_0.children = [c1, s3_2, s3_4]
        c1.parent = s2_0
        s3_2.parent = s2_0
        s3_4.parent = s2_0

        s2_1 = TreeNode(synset_='household')
        s2_1.base_class_id_list = [4, 8, 10, 15, 17, 19]
        s2_1.children = [c4, c15, c19, s3_3]
        c4.parent = s2_1
        c15.parent = s2_1
        c19.parent = s2_1
        s3_3.parent = s2_1

        s2_3 = TreeNode(synset_='vehicle')
        s2_3.base_class_id_list = [0, 2, 3, 5, 6, 13, 18]
        s2_3.children = [c0, c3, c18, s3_0, s3_1]
        c0.parent = s2_3
        c3.parent = s2_3
        c18.parent = s2_3
        s3_0.parent = s2_3
        s3_1.parent = s2_3

        # superclass 1st layer root #
        root = TreeNode(synset_='object')
        root.base_class_id_list = list(range(20))
        root.children = [c14, s2_0, s2_1, s2_3]
        s2_0.parent = root
        s2_1.parent = root
        c14.parent = root
        s2_3.parent = root

        tree = ClassTree()
        tree.root_node = root
        tree.leaf_node_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19]
        tree.base_class_node_list = tree.leaf_node_list
        tree.UpdateBSF_id()
        return tree

    @staticmethod
    def ConstructClassTree(class_inf: ClassSpaceInformation):
        tree = ClassTree()
        tree.class_inf = class_inf
        tree.__WordNet2Tree()
        # print('tree.__WordNet2Tree() end')
        tree.__PruneTree()
        # print('tree.__PruneTree() end')
        tree.UpdateBSF_id()
        # print('tree.UpdateBSF_id() end')
        return tree

    @staticmethod
    def BSF_WordNet(file_name=None):
        f_writer = None
        if file_name is not None:
            f_writer = open(file=file_name, mode='w')
            f_writer.write('ID\tsynset\n')
        root_node = wn.synset('entity.n.01')
        queue = List()
        queue.InQueue(root_node)
        i = 1
        while not queue.IsEmpty():
            parent_node = queue.OutQueue()
            if file_name is not None:
                f_writer.write(str(i) + '\t' + str(parent_node) + '\n')
            print(str(i) + ' ' + str(parent_node))
            i += 1

            hyponym_list = parent_node.hyponyms()
            if len(hyponym_list) > 0:
                for item in hyponym_list:
                    queue.InQueue(item)

        if file_name is not None:
            f_writer.close()
        return root_node

    @staticmethod
    def BFS_Tree(root_node: TreeNode, file_name=None):
        f_writer = None
        if file_name is not None:
            f_writer = open(file=file_name, mode='w')
            f_writer.write('BFS-ID\tNode.synset_\t base_class_num\t base_class_set\n')
        ClassTree.counter = 1
        queue = List()
        queue.InQueue(root_node)
        while not queue.IsEmpty():
            node = queue.OutQueue()
            if file_name is not None:
                f_writer.write(str(ClassTree.counter) + '\t' +
                               node.SynsetName() + '\t' +
                               str(node.base_class_id_list.__len__()) + '\t' +
                               node.BaseClassStr() + '\n')
            print(str(ClassTree.counter) + ', ' + node.SynsetName())
            ClassTree.counter += 1
            if not node.IsLeaf():
                for child_node in node.children:
                    queue.InQueue(child_node)

        if file_name is not None:
            f_writer.close()
