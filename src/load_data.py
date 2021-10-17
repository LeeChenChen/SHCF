import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import tensorflow as tf

class Data(object):
    def __init__(self, args, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.max_len = args.max_len

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        if args.sample:
            train_file = path + '/s_train.txt'
            test_file = path + '/s_test.txt'

        #get number of users and items
        self.n_users, self.n_items, self.max_seq = 0, 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        
        self.exist_users = []
        with open(train_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        self.exist_users.append(uid)
                        self.n_items = max(self.n_items, max(items))
                        self.max_seq = max(self.max_seq, len(items))
                        self.n_users = max(self.n_users, uid)
                        self.n_train += len(items)


        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.max_seq += 1

        self.print_statistics()
        
            
        self.adj_list=[]
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.Seq = np.zeros((self.n_users, self.n_items), dtype=np.int32)
        self.Seq_item = sp.dok_matrix((self.n_users, self.max_seq), dtype=np.int32)
        # self.Seq = sp.dok_matrix([self.n_users, self.max_seq], dtype=np.int32)
        # self.Seq_item = sp.dok_matrix([self.n_users, self.max_seq], dtype=np.int32)

        self.train_items, self.test_set, self.test_seq, self.test_mask = {}, {}, {}, {}
        with open(train_file) as f_train:
            
            for l in f_train.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]

                for ind,i in enumerate(train_items):
                    self.R[uid, i] = 1.
                    self.Seq[uid, i] = ind+1
                    self.Seq_item[uid, ind] = i

                self.train_items[uid] = train_items
        with open(test_file) as f_test:
            for l in f_test.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')]
                except Exception:
                    continue

                uid, test_items = items[0], items[1:]
                self.test_set[uid] = test_items
                if len(self.train_items[uid]) >= args.max_len:
                    self.test_seq[uid] = self.train_items[uid][-args.max_len:][::-1]
                    self.test_mask[uid] = [1.0] * args.max_len

                else:
                    self.test_seq[uid] = self.train_items[uid][::-1] + [0]*(args.max_len-len(self.train_items[uid]))
                    self.test_mask[uid] = [1.0]*len(self.train_items[uid]) + [0.0]*(args.max_len-len(self.train_items[uid]))
        
        if 'ml' in path:
            self.n_categary = 19
            self.n_city = 0
            type_list = ['user','item','categary']
            len_n = [0, self.n_users, self.n_users + self.n_items, self.n_users + self.n_items + self.n_categary]
        else:
            self.n_categary = 492
            self.n_city = 58
            type_list = ['user','item','categary','city']
            l_all = self.n_users + self.n_items + self.n_categary + self.n_city
            len_n = [0, self.n_users, self.n_users + self.n_items, self.n_users + self.n_items + self.n_categary, l_all]
            
        try:
            norm_adj_mat = sp.load_npz(self.path + '/norm_adj_mat.npz')
            
            print('already load adj matrix')
        except Exception:
            if 'ml' in path:
                cate = '/item_categary.txt'
                if args.sample:
                    cate = '/s_item_categary.txt'
                with open(path+cate) as file:
                    for l in file.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        item, categarys = items[0], items[1:]
                        self.C = sp.dok_matrix((self.n_items, self.n_categary), dtype=np.float32)
                        for i in categarys[1:]:
                            self.C[item, i] = 1.

                adj_mat = sp.dok_matrix((self.n_users+self.n_items+self.n_categary, self.n_users+self.n_items+self.n_categary), dtype=np.float32)
                adj_mat[:self.n_users, self.n_users:self.n_users+self.n_items] = self.R
                adj_mat[self.n_users:self.n_users+self.n_items, :self.n_users] = self.R.T
                adj_mat[self.n_users:self.n_users+self.n_items, self.n_users+self.n_items:] = self.C
                adj_mat[self.n_users+self.n_items:, self.n_users:self.n_users+self.n_items] = self.C.T

            else:
                cate = '/item_categary.txt'
                if args.sample:
                    cate = '/s_item_categary.txt'
                with open(path+cate) as file:
                    for l in file.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        item, categarys = items[0], items[1:]
                        self.C = sp.dok_matrix((self.n_items, self.n_categary), dtype=np.float32)
                        for i in categarys[1:]:
                            self.C[item, i] = 1.
                city = '/item_city.txt'
                if args.sample:
                    city = '/s_item_city.txt'
                with open(path + city) as file:
                    for l in file.readlines():
                        if len(l) == 0: break
                        l = l.strip('\n')
                        items = [int(i) for i in l.split(' ')]
                        item, citis = items[0], items[1:]
                        self.Ci = sp.dok_matrix((self.n_items, self.n_city), dtype=np.float32)
                        for i in citis[1:]:
                            self.Ci[item, i] = 1.

                adj_mat = sp.dok_matrix((l_all, l_all), dtype=np.float32)
                adj_mat[:self.n_users, self.n_users:self.n_users+self.n_items] = self.R
                adj_mat[self.n_users:self.n_users+self.n_items, :self.n_users] = self.R.T
                adj_mat[self.n_users:self.n_users+self.n_items, self.n_users+self.n_items:l_all-self.n_city] = self.C
                adj_mat[self.n_users:self.n_users + self.n_items, l_all-self.n_city:] = self.Ci
                adj_mat[self.n_users+self.n_items:l_all-self.n_city, self.n_users:self.n_users+self.n_items] = self.C.T
                adj_mat[l_all-self.n_city:, self.n_users:self.n_users + self.n_items] = self.Ci.T
            norm_adj_mat = self.normalize_adj(adj_mat + sp.eye(adj_mat.shape[0]))
            sp.save_npz(self.path + '/norm_adj_mat.npz', norm_adj_mat)
        self.adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]

        for i in range(len(type_list)):
            for j in range(len(type_list)):
                self.adj_list[i][j] = norm_adj_mat[len_n[i]:len_n[i+1], len_n[j]:len_n[j+1]].astype(np.float32)
            

    def normalize_adj(self, mx):
        """Row-normalize sparse matrix"""

        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            sequnce = []
            mask = []
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break

                pos_id = np.random.randint(low=1, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
                    if pos_id > self.max_len:
                        sequnce.append(pos_items[pos_id - self.max_len:pos_id][::-1])
                        mask.append([1.0]*self.max_len)
                    else:
                        sequnce.append(pos_items[:pos_id][::-1] + [0] * (self.max_len - pos_id))
                        mask.append([1.0]*pos_id + [0.0]*(self.max_len-pos_id))
            return pos_batch, sequnce, mask

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items, sequence, mask = [], [], [], []
        for u in users:
            p, s, m = sample_pos_items_for_u(u, 1)
            pos_items += p
            sequence += s
            mask += m
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items, sequence, mask

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d, max_seq=%d' % (self.n_users, self.n_items, self.max_seq))
        print('n_interactions=%d' % (self.n_train + self.n_items))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_items)/(self.n_users * self.n_items)))
