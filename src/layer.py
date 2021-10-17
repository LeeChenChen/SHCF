import numpy as np
import tensorflow as tf

def attn(args, feature_list, adj_list, pos_emb, pos_mat, layer):
    ntype = len(feature_list)
    new_feature_list = []
    for i in range(ntype):
        feature_i = []
        for j in range(ntype):
            _name_scope = 'layer' + str(layer) + '_self_attn' + str(i) + '_' + str(j)
            if i==j:
                feature_i_j = feature_list[i]

            else:
                # adj 是个零矩阵
                if adj_list[i][j].nnz == 0:
                    # continue
                    feature_i_j = tf.zeros([feature_list[i].shape[0], args.embed_size], dtype=tf.float32)
                else:
                    adj = _convert_sp_mat_to_sp_tensor(adj_list[i][j])
                    feature_i_j = self_attn(args, feature_list[i],feature_list[j], adj, _name_scope)
            # print(i,j,adj_list[i][j].nnz,feature_i_j)
            feature_i_j = tf.nn.dropout(feature_i_j, 1-eval(args.mess_dropout)[layer])
            feature_i_j = tf.math.l2_normalize(feature_i_j, axis=1)
            feature_i.append(feature_i_j)
        new_feature_list.append(second_attn(args, feature_i, 'second_attn_'+str(i)))
    # new_feature_list.append(feature_list[2])
    return new_feature_list




def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def mlp(w,x,b):
    return tf.nn.tanh(tf.nn.bias_add(tf.matmul(x,w),b))


def seq_attn(args, feature1, seq_item, pos_emb, mask):
    with tf.variable_scope('sequential_aware_attention', reuse=tf.AUTO_REUSE):
        seq = tf.nn.embedding_lookup(feature1, seq_item)
        mask = tf.expand_dims(mask, -1)
        seq *= mask
        pos = pos_emb*mask
        seq += pos

        Q = tf.layers.dense(seq, args.embed_size, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(seq, args.embed_size, activation=None)  # (N, T_k, C)
        V = tf.layers.dense(seq, args.embed_size, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, args.n_head, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, args.n_head, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, args.n_head, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(seq, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [args.n_head, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(seq, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [args.n_head, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(seq)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)


        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        h_dynamic = tf.reduce_sum(tf.concat(tf.split(outputs, args.n_head, axis=0), axis=2),1)  # (N, C)

        # static

        M = seq.shape.as_list()[1]

        initializer = tf.contrib.layers.xavier_initializer()
        w_static = tf.get_variable('w_static', initializer=initializer([M, 1])) #[M,1]
        s_feature2 = tf.transpose(seq, perm=[0, 2, 1]) #[N,F,M]
        coef = tf.reshape(tf.squeeze(tf.matmul(tf.reshape(s_feature2,[-1, M]), w_static)),[-1, args.embed_size]) #[N,F]
        belta = tf.expand_dims(tf.nn.tanh(coef),-1)  #[N,F,1]
        h_static = tf.reduce_sum(tf.multiply(belta, s_feature2),-1)

        h_prime = args.lamb * h_dynamic + (1-args.lamb) * h_static #[N,F]
        return h_prime


def self_attn(args, feature1, feature2, adj, name_scope):
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        initializer = tf.contrib.layers.xavier_initializer()
        w1 = tf.get_variable('w1', initializer=initializer([args.embed_size, 1]))
        b1 = tf.get_variable('b1', initializer=initializer([1]))
        w2 = tf.get_variable('w2', initializer=initializer([args.embed_size, 1]))
        b2 = tf.get_variable('b2', initializer=initializer([1]))
        h = feature1
        g = feature2
        N = h.shape.as_list()[0]
        M = g.shape.as_list()[0]

        e1 = tf.tile(mlp(w1, h, b1), [1, M])
        e2 = tf.transpose(tf.tile(mlp(w2, g, b2), [1, N]))
        e = e1 + e2

        e = adj * e
        attention = tf.SparseTensor(indices=e.indices,
                                values=tf.nn.leaky_relu(e.values),
                                dense_shape=e.dense_shape)
        attention = tf.sparse.to_dense(attention)
        zero_vec = -9e15 * tf.ones_like(attention)

        attention = tf.where(tf.sparse.to_dense(adj) > 0, attention, zero_vec)
        attention = tf.nn.softmax(attention, axis=1)

        h_prime = tf.matmul(attention, g)
        return h_prime


def second_attn(args, inputs, name_scope):
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        w_omega = tf.get_variable('w_omega', initializer=tf.random_normal([args.embed_size, 1], stddev=0.1))
        b_omega = tf.get_variable('b_omega', initializer=tf.random_normal([1], stddev=0.1))
        u_omega = tf.get_variable('o_omega', initializer=tf.random_normal([1], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return output