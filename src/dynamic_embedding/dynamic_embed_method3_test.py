'''
method 3
weight is context
'''

import numpy as np
import tensorflow as tf
import math

from utils.data_handler import DataHandler as dh

class NodeEmbedding(object):
    def __init__(self, params, init_embeddings, init_weights, G, mapp, rmapp, num_to_del = 1):
        print("dynamic embed initiating:")
        self.num_nodes_init, self.embedding_size = init_embeddings.shape
        self.num_remain = G.number_of_nodes() - num_to_del
        self.num_to_del = num_to_del


        self.num_modify = params["num_modify"]
        self.batch_size = params["batch_size"]
        self.learn_rate = params["learn_rate"]
        self.optimizer = params["optimizer"] if "optimizer" in params else "GradientDescentOptimizer"
        self.tol = params["tol"] if "tol" in params else 0.0001
        self.neighbor_size = params["neighbor_size"]
        self.negative_distortion = params["negative_distortion"]
        self.num_sampled = params["num_sampled"]
        self.epoch_num = params["epoch_num"]
        self.lbd_batch = params["lambda_batch"]
        self.lbd_all = params["lambda_all"]

        self.bs = __import__("batch_strategy." + params["batch_strategy"]["func"], fromlist = ["batch_strategy"]).BatchStrategy(G, num_to_del, mapp, rmapp, self.num_modify, params["batch_strategy"])

        unigrams_in = None
        if "in_negative_sampling_distribution" in params:
            unigrams_in = getattr(dh, params["in_negative_sampling_distribution"]["func"])(G, params["in_negative_sampling_distribution"], self.num_to_del)
        unigrams_out = None
        if "out_negative_sampling_distribution" in params:
            unigrams_out = getattr(dh, params["out_negative_sampling_distribution"]["func"])(G, params["out_negative_sampling_distribution"], self.num_to_del)

        self.tensor_graph = tf.Graph()

        with self.tensor_graph.as_default():
            tf.set_random_seed(157)
            self.constant_embeddings = tf.constant(init_embeddings[: self.num_remain - self.num_modify])
            self.constant_embeddings_to_del = tf.constant(init_embeddings[self.num_remain: self.num_nodes_init])
            self.constant_weights = tf.constant(init_weights[: self.num_remain - self.num_modify])
            self.weights_to_del = tf.Variable(tf.random_uniform([self.num_to_del, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            self.weights_to_del_pre = tf.constant(init_weights[self.num_remain: self.num_nodes_init])
            self.modify_embeddings_pre = tf.constant(init_embeddings[self.num_remain - self.num_modify: self.num_remain])
            self.modify_weights_pre = tf.constant(init_weights[self.num_remain - self.num_modify: self.num_remain])
            self.modify_embeddings = tf.Variable(tf.random_uniform([self.num_modify, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            self.modify_weights = tf.Variable(tf.random_uniform([self.num_modify, self.embedding_size], -1.0, 1.0), dtype = tf.float32)
            # self.modify_embeddings = tf.Variable(init_embeddings[self.num_remain-self.num_modify: self.num_remain],dtype = tf.float32)
            # self.modify_weights = tf.Variable(init_weights[self.num_remain-self.num_modify: self.num_remain], dtype = tf.float32)
            #self.modify_embeddings = tf.Variable(self.modify_embeddings_pre)
            #self.modify_weights = tf.Variable(self.modify_weights_pre)

            self.x_in = tf.placeholder(tf.int64, shape = [None])
            self.x_out = tf.placeholder(tf.int64, shape = [None])
            self.x_in_neg = tf.placeholder(tf.int64, shape=[None])
            self.x_out_neg = tf.placeholder(tf.int64, shape=[None])
            self.labels_in = tf.placeholder(tf.int64, shape = [None, self.neighbor_size])
            self.labels_out = tf.placeholder(tf.int64, shape = [None, self.neighbor_size])
            self.labels_in_neg = tf.placeholder(tf.int64, shape=[None, self.neighbor_size])
            self.labels_out_neg = tf.placeholder(tf.int64, shape=[None, self.neighbor_size])

            self.nce_biases = tf.zeros([self.num_nodes_init], tf.float32)

            self.embed = tf.concat([self.constant_embeddings, self.modify_embeddings, self.constant_embeddings_to_del], 0)
            self.w = tf.concat([self.constant_weights, self.modify_weights, self.weights_to_del], 0)

            self.delta_embeddings_pad = tf.concat(
                    [tf.zeros([self.num_remain - self.num_modify, self.embedding_size], dtype = tf.float32),
                        self.modify_embeddings - self.modify_embeddings_pre,
                        tf.zeros([self.num_to_del, self.embedding_size], dtype = tf.float32)],
                    axis = 0)
            self.delta_weights_pad = tf.concat(
                    [tf.zeros([self.num_remain - self.num_modify, self.embedding_size], dtype = tf.float32),
                        self.modify_weights - self.modify_weights_pre,
                        self.weights_to_del - self.weights_to_del_pre],
                    axis = 0)

            self.embedding_batch = tf.nn.embedding_lookup(self.embed, self.x_in)
            self.embedding_batch_neg = tf.nn.embedding_lookup(self.embed, self.x_in_neg)
            self.weight_batch = tf.nn.embedding_lookup(self.w, self.x_out)
            self.weight_batch_neg = tf.nn.embedding_lookup(self.w, self.x_out_neg)
            self.delta_embeddings_batch = tf.nn.embedding_lookup(self.delta_embeddings_pad, self.x_in)
            self.delta_weights_batch = tf.nn.embedding_lookup(self.delta_weights_pad, self.x_out)

            if unigrams_in is None:
                self.loss_in = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.w,
                        biases = self.nce_biases,
                        labels = self.labels_in,
                        inputs = self.embedding_batch,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes_init,
                        num_true = self.neighbor_size))
                self.loss_in_neg = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.w,
                        biases = self.nce_biases,
                        labels = self.labels_in_neg,
                        inputs = self.embedding_batch_neg,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes_init,
                        num_true = self.neighbor_size))
            else:
                self.sampled_values_in = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes = self.labels_in,
                    num_true = self.neighbor_size,
                    num_sampled = self.num_sampled,
                    unique = False,
                    range_max = self.num_remain,
                    distortion = self.negative_distortion,
                    unigrams = unigrams_in)
                self.loss_in = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.w,
                        biases=self.nce_biases,
                        labels=self.labels_in,
                        inputs=self.embedding_batch,
                        num_sampled=self.num_sampled,
                        num_classes=self.num_nodes_init,
                        num_true=self.neighbor_size,
                        sampled_values=self.sampled_values_in
                    ))
                self.sampled_values_in_neg = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=self.labels_in_neg,
                    num_true=self.neighbor_size,
                    num_sampled=self.num_sampled,
                    unique=False,
                    range_max=self.num_remain,
                    distortion=self.negative_distortion,
                    unigrams=unigrams_in)
                self.loss_in_neg = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.w,
                        biases=self.nce_biases,
                        labels=self.labels_in_neg,
                        inputs=self.embedding_batch_neg,
                        num_sampled=self.num_sampled,
                        num_classes=self.num_nodes_init,
                        num_true=self.neighbor_size,
                        sampled_values=self.sampled_values_in_neg
                    ))

            if unigrams_out is None:
                self.loss_out = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.embed,
                        biases = self.nce_biases,
                        labels = self.labels_out,
                        inputs = self.weight_batch,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes_init,
                        num_true = self.neighbor_size))
                self.loss_out_neg = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.embed,
                        biases = self.nce_biases,
                        labels = self.labels_out_neg,
                        inputs = self.weight_batch_neg,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes_init,
                        num_true = self.neighbor_size))
            else:
                self.sampled_values_out = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes = self.labels_out,
                    num_true = self.neighbor_size,
                    num_sampled = self.num_sampled,
                    unique = False,
                    range_max = self.num_remain,
                    distortion = self.negative_distortion,
                    unigrams = unigrams_out)
                self.loss_out = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights = self.embed,
                        biases = self.nce_biases,
                        labels = self.labels_out,
                        inputs = self.weight_batch,
                        num_sampled = self.num_sampled,
                        num_classes = self.num_nodes_init,
                        num_true = self.neighbor_size,
                        sampled_values = self.sampled_values_out))
                self.sampled_values_out_neg = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=self.labels_out_neg,
                    num_true=self.neighbor_size,
                    num_sampled=self.num_sampled,
                    unique=False,
                    range_max=self.num_remain,
                    distortion=self.negative_distortion,
                    unigrams=unigrams_out)
                self.loss_out_neg = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.embed,
                        biases=self.nce_biases,
                        labels=self.labels_out_neg,
                        inputs=self.weight_batch_neg,
                        num_sampled=self.num_sampled,
                        num_classes=self.num_nodes_init,
                        num_true=self.neighbor_size,
                        sampled_values=self.sampled_values_out_neg))

            #self.loss = self.loss_out + self.loss_in - self.loss_out_neg - self.loss_in_neg + self.lbd * (tf.norm(self.delta_embeddings_batch) + tf.norm(self.delta_weights_batch))
            #self.loss = self.loss_out + self.loss_in + self.loss_out_neg + self.loss_in_neg + self.lbd * (tf.norm(self.delta_embeddings_batch) + tf.norm(self.delta_weights_batch))
            self.loss_pos = self.loss_out + self.loss_in
            self.loss_neg = self.loss_in_neg
            self.loss_reg_batch = self.lbd_batch * (
                        tf.norm(self.delta_embeddings_batch) + tf.norm(self.delta_weights_batch))
            self.loss_reg_all = self.lbd_all * (tf.norm(self.delta_embeddings_pad) + tf.norm(self.delta_weights_pad))
            #self.loss = self.loss_pos - tf.nn.sigmoid(self.loss_neg) + self.loss_reg
            self.loss = self.loss_pos - self.loss_neg + self.loss_reg_batch + self.loss_reg_all
            self.train_step = getattr(tf.train, self.optimizer)(self.learn_rate).minimize(self.loss)
            print("dynamic embed initiate done")

    def train(self, save_path = None):

        print("(test)dynamic embed train begins:")

        with tf.Session(graph = self.tensor_graph) as sess:
            sess.run(tf.global_variables_initializer())
            print("dynamic embed train begin")
            # print("self.embed", sess.run(self.embed))
            # print("self.w", sess.run(self.embed))

            # print("dynamic embed train get batch begin:")
            # batch_x_in, batch_x_out, batch_labels_in, batch_labels_out, batch_x_in_neg, batch_x_out_neg, batch_labels_in_neg, batch_labels_out_neg = self.bs.get_batch(
            #     self.batch_size)
            # print("dynamic embed train get_batch done")

            for i in xrange(self.epoch_num):
                batch_x_in, batch_x_out, batch_labels_in, batch_labels_out,batch_x_in_neg,batch_x_out_neg,batch_labels_in_neg,batch_labels_out_neg = self.bs.get_batch(self.batch_size)
                self.train_step.run(
                    {
                    self.x_in : batch_x_in,
                    self.x_out : batch_x_out,
                    self.labels_in : batch_labels_in,
                    self.labels_out : batch_labels_out,
                    self.x_in_neg: batch_x_in_neg,
                    self.x_out_neg: batch_x_out_neg,
                    self.labels_in_neg: batch_labels_in_neg,
                    self.labels_out_neg: batch_labels_out_neg
                    })
                # if (i == 1):
                #     loss = self.loss.eval(
                #         {
                #             self.x_in: batch_x_in,
                #             self.x_out: batch_x_out,
                #             self.labels_in: batch_labels_in,
                #             self.labels_out: batch_labels_out,
                #             self.x_in_neg: batch_x_in_neg,
                #             self.x_out_neg: batch_x_out_neg,
                #             self.labels_in_neg: batch_labels_in_neg,
                #             self.labels_out_neg: batch_labels_out_neg
                #         })
                #     print("i==1:")
                #     print(loss)
                if (i % 1000 == 0):
                    loss = self.loss.eval(
                        {
                            self.x_in: batch_x_in,
                            self.x_out: batch_x_out,
                            self.labels_in: batch_labels_in,
                            self.labels_out: batch_labels_out,
                            self.x_in_neg: batch_x_in_neg,
                            self.x_out_neg: batch_x_out_neg,
                            self.labels_in_neg: batch_labels_in_neg,
                            self.labels_out_neg: batch_labels_out_neg
                        })
                    print("loss:"+str(loss))

                    loss_out = self.loss_out.eval(
                        {
                            self.x_in: batch_x_in,
                            self.x_out: batch_x_out,
                            self.labels_in: batch_labels_in,
                            self.labels_out: batch_labels_out,
                            self.x_in_neg: batch_x_in_neg,
                            self.x_out_neg: batch_x_out_neg,
                            self.labels_in_neg: batch_labels_in_neg,
                            self.labels_out_neg: batch_labels_out_neg
                        })
                    print("loss_out:"+str(loss_out))

                    loss_out_neg = self.loss_out_neg.eval(
                        {
                            self.x_in: batch_x_in,
                            self.x_out: batch_x_out,
                            self.labels_in: batch_labels_in,
                            self.labels_out: batch_labels_out,
                            self.x_in_neg: batch_x_in_neg,
                            self.x_out_neg: batch_x_out_neg,
                            self.labels_in_neg: batch_labels_in_neg,
                            self.labels_out_neg: batch_labels_out_neg
                        })
                    print("loss_out_neg:" + str(loss_out_neg))

                    loss_in = self.loss_in.eval(
                        {
                            self.x_in: batch_x_in,
                            self.x_out: batch_x_out,
                            self.labels_in: batch_labels_in,
                            self.labels_out: batch_labels_out,
                            self.x_in_neg: batch_x_in_neg,
                            self.x_out_neg: batch_x_out_neg,
                            self.labels_in_neg: batch_labels_in_neg,
                            self.labels_out_neg: batch_labels_out_neg
                        })
                    print("loss_in:" + str(loss_in))

                    loss_in_neg = self.loss_in_neg.eval(
                        {
                            self.x_in: batch_x_in,
                            self.x_out: batch_x_out,
                            self.labels_in: batch_labels_in,
                            self.labels_out: batch_labels_out,
                            self.x_in_neg: batch_x_in_neg,
                            self.x_out_neg: batch_x_out_neg,
                            self.labels_in_neg: batch_labels_in_neg,
                            self.labels_out_neg: batch_labels_out_neg
                        })
                    print("loss_in_neg:" + str(loss_in_neg))
                    #
                    # self.norm_val = self.lbd * (tf.norm(self.delta_embeddings_batch) + tf.norm(self.delta_weights_batch))
                    # norm_val = self.norm_val.eval(
                    #     {
                    #         self.x_in: batch_x_in,
                    #         self.x_out: batch_x_out,
                    #         self.labels_in: batch_labels_in,
                    #         self.labels_out: batch_labels_out,
                    #         self.x_in_neg: batch_x_in_neg,
                    #         self.x_out_neg: batch_x_out_neg,
                    #         self.labels_in_neg: batch_labels_in_neg,
                    #         self.labels_out_neg: batch_labels_out_neg
                    #     })
                    # print("norm_val:"+str(norm_val))
                    # embedding_batch_neg = self.embedding_batch_neg.eval({
                    #     self.x_in_neg: batch_x_in_neg
                    # })
                    # print("batch_x_in_neg:")
                    # print(batch_x_in_neg)
                    # print("embedding_batch_neg:")
                    # print(embedding_batch_neg)
                    #
                    # labels_in_neg = self.labels_in_neg.eval({
                    #     self.labels_in_neg: batch_labels_in_neg
                    # })
                    # print("batch_labels_in_neg:")
                    # print(batch_labels_in_neg)
                    # print("labels_in_neg:")
                    # print(labels_in_neg)




            # print("printbatch:")
            # print("batch_x_in:",batch_x_in)
            # print("batch_x_out:", batch_x_out)
            # print("batch_labels_in:", batch_labels_in)
            # print("batch_labels_out:", batch_labels_out)
            # print("batch_x_in_neg:", batch_x_in_neg)
            # print("batch_x_out_neg", batch_x_out_neg)
            # print("batch_labels_in_neg:", batch_labels_in_neg)
            # print("batch_labels_out_neg:", batch_labels_out_neg)

            # print("use batch to get loss without train:")
            # loss = self.loss.eval(
            #     {
            #         self.x_in: batch_x_in,
            #         self.x_out: batch_x_out,
            #         self.labels_in: batch_labels_in,
            #         self.labels_out: batch_labels_out,
            #         self.x_in_neg: batch_x_in_neg,
            #         self.x_out_neg: batch_x_out_neg,
            #         self.labels_in_neg: batch_labels_in_neg,
            #         self.labels_out_neg: batch_labels_out_neg
            #     })
            # print("loss without train:")
            # print(loss)

            print("dynamic embed train return")

            return sess.run(self.embed), sess.run(self.w)

