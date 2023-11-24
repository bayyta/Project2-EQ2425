from asyncore import read
from os import listdir
from os.path import join
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans as MB_KMeans
import re
from tqdm import tqdm
from timeit import default_timer as timer


def read_filenames(folder, file_type='JPG'):
    """ Returns list of image paths. """
    imgs = [join(folder, f) for f in listdir(folder) if f.endswith(file_type)]
    return imgs

class Node:
    """ Class to represent a node in the vocabulary tree. """
    def __init__(self, index=None, mean=None, weight=None):
        self.children = []
        self.index = index
        self.mean = mean

class VocabularyTree:
    def __init__(self, contrast_threshold=0.03, edge_threshold=6.0, qy_img_folder='./Data2/client', db_img_folder='./Data2/server', qy_features_path='./query_features.npz', db_features_path='./database_features.npz'):
        self.num_objects = 50
        self.qy_features_path, self.db_features_path = qy_features_path, db_features_path
        self.qy_img_folder, self.db_img_folder = qy_img_folder, db_img_folder
        # Best: contrast_threshold=0.03, edge_threshold=6.0
        self.detector = cv.xfeatures2d.SIFT_create(nfeatures=400, contrastThreshold = contrast_threshold, edgeThreshold = edge_threshold)

        # Index
        self.qy_path_to_ind, self.qy_ind_to_path, self.qy_ind_to_obj = self.index_imgs(read_filenames(self.qy_img_folder))
        self.db_path_to_ind, self.db_ind_to_path, self.db_ind_to_obj = self.index_imgs(read_filenames(self.db_img_folder))
        self.num_qy_imgs = len(self.qy_ind_to_path)
        self.num_db_imgs = len(self.db_ind_to_path)

    def index_imgs(self, paths):
        path_to_ind, ind_to_path, ind_to_obj = {}, [], []
        temp = []
        for path in paths:
            inds = re.findall(r'\d+', path)[-2:]
            obj_ind = int(inds[0]) - 1
            obj_repl = int(inds[1]) - 1
            temp.append([path, obj_ind, obj_repl])
        temp.sort(key=lambda x: x[2])
        temp.sort(key=lambda x: x[1])
        for i, img in enumerate(temp):
            path_to_ind[img[0]] = i
            ind_to_path.append(img[0])
            ind_to_obj.append(img[1])

        return path_to_ind, ind_to_path, ind_to_obj

    def get_object_from_path(self, path):
        return int(re.findall(r'\d+', path)[-2])
        
    def compute_features(self):
        """ Load images and compute SIFT features to store. """
        qy_img_paths = read_filenames(self.qy_img_folder)
        db_img_paths = read_filenames(self.db_img_folder)

        self.qy_features = self._compute_and_save_features(qy_img_paths, self.qy_features_path, self.qy_path_to_ind)
        self.db_features = self._compute_and_save_features(db_img_paths, self.db_features_path, self.db_path_to_ind)

    def _compute_and_save_features(self, img_paths, save_path, ind_mapping):
        # Map features to index of image
        features = []
        tot_desc = 0
        for path in tqdm(img_paths, desc=f"Computing SIFT features to '{save_path}'"):
            img = cv.imread(path)
            _, des = self.detector.detectAndCompute(img, None)
            tot_desc += des.shape[0]
            features.append([des, ind_mapping[path]])
        tot_desc /= len(img_paths)
        print(f"Avg features: {int(tot_desc)}")

        """ Sorting here is slow, might change """
        features.sort(key=lambda x: x[1])
        features = [d[0] for d in features]

        np.savez(save_path, *features)
        return features

    def load_features(self):
        """ Load SIFT features from file. """
        qy_npz = np.load(self.qy_features_path)
        self.qy_features = [v for _, v in qy_npz.items()]

        db_npz = np.load(self.db_features_path)
        self.db_features = [v for _, v in db_npz.items()]

    def hi_kmeans(self, data=None, b=4, depth=3):
        """ Initialize root node and run _build_tree recursively. """
        print(f"Building tree (branch: {b}, depth: {depth})...", end=" ")
        start = timer()
        if data is None:
            points = np.array([feat for l1 in self.db_features for feat in l1])
        else:
            points = data
        self.root = Node(index=0)
        self.num_nodes = 1
        self.leaf_nodes_inds = []
        self.preorder = [[np.zeros(128, dtype=float)], [self.root.index]] # Save preorder traversal of tree [root, left, right]
        self._build_tree(self.root, points, b, depth, 0)
        end = timer()
        print(f"done (elapsed time: {(end-start):.2f}s). ")

        # Calculate idf-values for nodes
        table = self.propagate_through_tree(self.db_features, self.num_db_imgs)

        N_i = table > 0
        N_i_sum = N_i.sum(axis=0, dtype=float)
        N_i_sum[np.where(N_i_sum==0)] = self.num_db_imgs
        self.idf = np.log(self.num_db_imgs / N_i_sum)
        self.preorder.append(self.idf)
        self.preorder.append(self.leaf_nodes_inds)

        # Calculate tf value for database images
        tf = np.zeros((self.num_db_imgs, len(self.leaf_nodes_inds)))
        tf[:, :] = table[:, self.leaf_nodes_inds]
        tf /= (tf.sum(axis=1)).reshape((-1, 1))

        # Calculate tf-idf weight table for database
        self.tf_idf = tf * self.idf[self.leaf_nodes_inds]
        self.preorder.append(self.tf_idf)

        np.savez('./tree.npz', *self.preorder)

    def _build_tree(self, root, points, branch, depth, cur_depth):
        if cur_depth == depth:
            self.leaf_nodes_inds.append(root.index)
            return

        # Use mini-batch k-means if no. of points > 10,000
        if points.shape[0] > 10000:
            kmeans = MB_KMeans(n_clusters=branch, random_state=0).fit(points)    
        else:
            kmeans = KMeans(n_clusters=branch, random_state=0).fit(points)

        for i in range(branch):
            labels = [j for j in range(len(kmeans.labels_)) if kmeans.labels_[j] == i]
            child = Node(mean=kmeans.cluster_centers_[i], index=self.num_nodes)
            self.num_nodes += 1
            #print(self.num_nodes)
            root.children.append(child)
            self.preorder[0].append(child.mean)
            self.preorder[1].append(child.index)

            if len(labels) < branch and (cur_depth+1) < depth:
                # Just assign each cluster mean to the points
                for p in points[labels]:
                    c = Node(mean=p, index=self.num_nodes)
                    self.num_nodes += 1
                    #print(self.num_nodes)
                    child.children.append(c)
                    self.preorder[0].append(c.mean)
                    self.preorder[1].append(c.index)
                    self.leaf_nodes_inds.append(c.index)

                for _ in range(branch - len(labels)):
                    rand_mean = np.zeros(128) - 100000000.0
                    c = Node(mean=rand_mean, index=self.num_nodes)
                    self.num_nodes += 1
                    #print(self.num_nodes)
                    child.children.append(c)
                    self.preorder[0].append(c.mean)
                    self.preorder[1].append(c.index)
                    self.leaf_nodes_inds.append(c.index)
            else:
                self._build_tree(child, points[labels], branch, depth, cur_depth + 1)
        
    def load_tree(self, b=4, depth=3):
        """ Load preorder traversed tree recursively with _load_tree. """
        print(f"Branch: {b}, depth: {depth}")
        tree_npz = np.load('./tree.npz')
        self.tree_means, self.tree_inds = tree_npz['arr_0'].tolist(), tree_npz['arr_1'].tolist()
        self.idf, self.leaf_nodes_inds, self.tf_idf = tree_npz['arr_2'], tree_npz['arr_3'], tree_npz['arr_4']

        self.num_nodes = len(self.tree_means)
        self.root = Node()
        self._load_tree(self.root, b, depth, 0)

    def _load_tree(self, root, branch, depth, cur_depth):
        if root.mean is None:
            root.mean = np.array(self.tree_means.pop(0))
            root.index = self.tree_inds.pop(0)
        if cur_depth == depth or root.index in self.leaf_nodes_inds:
            return
        for _ in range(branch):
            node = Node()
            root.children.append(node)
            self._load_tree(node, branch, depth, cur_depth + 1)

    def propagate_through_tree(self, features, num_imgs):
        """ Compute f_ij: no. of features that go through each node in tree. """
        table = np.full((num_imgs, self.num_nodes), 0, dtype=int)
        for i, feats in tqdm(enumerate(features), desc="Propagating tree", total=num_imgs):
            for feat in feats:
                table[i, self.root.index] += 1 # All features go through root node
                stack = []
                stack.extend(self.root.children)
                while len(stack) > 0:
                    min_dist, min_node = float('inf'), None
                    for node in stack:
                        dist = np.linalg.norm(node.mean - feat)
                        if dist < min_dist:
                            min_dist = dist
                            min_node = node
                    table[i, min_node.index] += 1
                    stack.clear()
                    stack.extend(min_node.children)
        return table

    def query_image(self, qy_imgs=None, num_matches=5, fraction=1):
        if qy_imgs is None:
            qy_imgs = np.arange(self.num_qy_imgs) # Use all query images

        if fraction < 1:
            qy_features = []
            for i, feats in enumerate(self.qy_features):
                num_features = int(fraction * feats.shape[0])
                feat_inds = np.arange(feats.shape[0])
                np.random.shuffle(feat_inds)
                qy_features.append(feats[feat_inds[:num_features], :])
        else:
            qy_features = self.qy_features

        qy_table = self.propagate_through_tree(qy_features, self.num_qy_imgs)
        queries = np.zeros((len(qy_imgs), len(self.leaf_nodes_inds)))
        queries[:, :] = qy_table[qy_imgs[:, None], self.leaf_nodes_inds]
        queries /= (queries.sum(axis=1)).reshape((-1, 1))
        queries *= self.idf[self.leaf_nodes_inds]

        results = []
        top_1_recall, top_5_recall = 0, 0
        for i in range(len(qy_imgs)):
            query = queries[i, :]
            query_img_path = self.qy_ind_to_path[qy_imgs[i]]

            diff_l1 = np.abs(self.tf_idf - query)
            scores = diff_l1.sum(axis=1)

            ranking = np.argsort(scores)
            results.append(ranking[0:num_matches])
            path_results = [self.db_ind_to_path[k] for k in results[i]]
            qy_obj = self.get_object_from_path(query_img_path)

            # Top 1 recall
            res_obj = self.get_object_from_path(path_results[0])
            if qy_obj == res_obj:
                top_1_recall += 1

            # Top 5 recall
            for res in path_results:
                res_obj = self.get_object_from_path(res)
                if qy_obj == res_obj:
                    top_5_recall += 1
                    break

            #print(f"Query {query_img_path} matches: {path_results}")
        top_1_recall /= len(qy_imgs)
        top_5_recall /= len(qy_imgs)
        print(f"Top-1 recall: {(100.0 * top_1_recall):.2f}%")
        print(f"Top-5 recall: {(100.0 * top_5_recall):.2f}%")



tree = VocabularyTree()
#tree.compute_features()
tree.load_features()
tree.hi_kmeans(b=5, depth=7)
#tree.load_tree(b=5, depth=7)
tree.query_image(fraction=0.5)

