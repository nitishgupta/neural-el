import os
import numpy as np
import tensorflow as tf

from models.base import Model


class LossOptim(object):
    def __init__(self, figermodel):
        ''' Houses utility functions to facilitate training/pre-training'''

        # Object of the WikiELModel Class
        self.figermodel = figermodel

    def make_loss_graph(self):
        self.figermodel.labeling_model.loss_graph(
          true_label_ids=self.figermodel.labels_batch,
          scope_name=self.figermodel.labeling_loss_scope,
          device_gpu=self.figermodel.device_placements['gpu'])

        self.figermodel.posterior_model.loss_graph(
          true_entity_ids=self.figermodel.true_entity_ids,
          scope_name=self.figermodel.posterior_loss_scope,
          device_gpu=self.figermodel.device_placements['gpu'])

        if self.figermodel.useCNN:
            self.figermodel.wikidescmodel.loss_graph(
              true_entity_ids=self.figermodel.true_entity_ids,
              scope_name=self.figermodel.wikidesc_loss_scope,
              device_gpu=self.figermodel.device_placements['gpu'])

    def optimizer(self, optimizer_name, name):
        if optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(
              learning_rate=self.figermodel.learning_rate,
              name='Adam_'+name)
        elif optimizer_name == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
              learning_rate=self.figermodel.learning_rate,
              name='Adagrad_'+name)
        elif optimizer_name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
              learning_rate=self.figermodel.learning_rate,
              name='Adadelta_'+name)
        elif optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
              learning_rate=self.figermodel.learning_rate,
              name='SGD_'+name)
        elif optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
              learning_rate=self.figermodel.learning_rate,
              momentum=0.9,
              name='Momentum_'+name)
        else:
            print("OPTIMIZER WRONG. HOW DID YOU GET HERE!!")
            sys.exit(0)
        return optimizer

    def weight_regularization(self, trainable_vars):
        vars_to_regularize = []
        regularization_loss = 0
        for var in trainable_vars:
            if "_weights" in var.name:
                regularization_loss += tf.nn.l2_loss(var)
                vars_to_regularize.append(var)

        print("L2 - Regularization for Variables:")
        self.figermodel.print_variables_in_collection(vars_to_regularize)
        return regularization_loss

    def label_optimization(self, trainable_vars, optim_scope):
        # Typing Loss
        if self.figermodel.typing:
            self.labeling_loss = self.figermodel.labeling_model.labeling_loss
        else:
            self.labeling_loss = tf.constant(0.0)

        if self.figermodel.entyping:
            self.entity_labeling_loss = \
                self.figermodel.labeling_model.entity_labeling_loss
        else:
            self.entity_labeling_loss = tf.constant(0.0)

        # Posterior Loss
        if self.figermodel.el:
            self.posterior_loss = \
                self.figermodel.posterior_model.posterior_loss
        else:
            self.posterior_loss = tf.constant(0.0)

        if self.figermodel.useCNN:
            self.wikidesc_loss = self.figermodel.wikidescmodel.wikiDescLoss
        else:
            self.wikidesc_loss = tf.constant(0.0)

        # _ = tf.scalar_summary("loss_typing", self.labeling_loss)
        # _ = tf.scalar_summary("loss_posterior", self.posterior_loss)
        # _ = tf.scalar_summary("loss_wikidesc", self.wikidesc_loss)

        self.total_loss = (self.labeling_loss + self.posterior_loss +
                           self.wikidesc_loss + self.entity_labeling_loss)

        # Weight Regularization
        # self.regularization_loss = self.weight_regularization(
        #   trainable_vars)
        # self.total_loss += (self.figermodel.reg_constant *
        #                     self.regularization_loss)

        # Scalar Summaries
        # _ = tf.scalar_summary("loss_regularized", self.total_loss)
        # _ = tf.scalar_summary("loss_labeling", self.labeling_loss)

        with tf.variable_scope(optim_scope) as s, \
            tf.device(self.figermodel.device_placements['gpu']) as d:
            self.optimizer = self.optimizer(
              optimizer_name=self.figermodel.optimizer, name="opt")
            self.gvs = self.optimizer.compute_gradients(
              loss=self.total_loss, var_list=trainable_vars)
            # self.clipped_gvs = self.clip_gradients(self.gvs)
            self.optim_op = self.optimizer.apply_gradients(self.gvs)

    def clip_gradients(self, gvs):
        clipped_gvs = []
        for (g,v) in gvs:
            if self.figermodel.embeddings_scope in v.name:
                clipped_gvalues = tf.clip_by_norm(g.values, 30)
                clipped_index_slices = tf.IndexedSlices(
                  values=clipped_gvalues,
                  indices=g.indices)
                clipped_gvs.append((clipped_index_slices, v))
            else:
                clipped_gvs.append((tf.clip_by_norm(g, 1), v))
        return clipped_gvs
