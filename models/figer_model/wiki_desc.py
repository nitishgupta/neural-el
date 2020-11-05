import time
import numpy as np
import tensorflow as tf

from models.base import Model

class WikiDescModel(Model):
    '''
    Input is sparse tensor of mention strings in mention's document.
    Pass through feed forward and get a coherence representation
    (keep same as context_encoded_dim)
    '''

    def __init__(self, desc_batch, trueentity_embs, negentity_embs, allentity_embs,
                 batch_size, doclength, wordembeddim, filtersize, desc_encoded_dim,
                 scope_name, device, dropout_keep_prob=1.0):
        """
        Parameters ---------- descents : int.

        Args:
            self: (todo): write your description
            desc_batch: (str): write your description
            trueentity_embs: (todo): write your description
            negentity_embs: (todo): write your description
            allentity_embs: (todo): write your description
            batch_size: (int): write your description
            doclength: (int): write your description
            wordembeddim: (str): write your description
            filtersize: (int): write your description
            desc_encoded_dim: (str): write your description
            scope_name: (str): write your description
            device: (todo): write your description
            dropout_keep_prob: (str): write your description
        """

        # [B, doclength, wordembeddim]
        self.desc_batch = desc_batch
        self.batch_size = batch_size
        self.doclength = doclength
        self.wordembeddim = wordembeddim
        self.filtersize = filtersize
        self.desc_encoded_dim = desc_encoded_dim  # Output dim of desc
        self.dropout_keep_prob = dropout_keep_prob

        # [B, K] - target of the CNN network and Negative sampled Entities
        self.trueentity_embs = trueentity_embs
        self.negentity_embs = negentity_embs
        self.allentity_embs = allentity_embs

        # [B, DL, WD, 1] - 1 to specify one channel
        self.desc_batch_expanded = tf.expand_dims(self.desc_batch, -1)
        # [F, WD, 1, K]
        self.filter_shape = [self.filtersize, self.wordembeddim, 1, self.desc_encoded_dim]


        with tf.variable_scope(scope_name) as scope, tf.device(device) as device:
            W = tf.Variable(tf.truncated_normal(self.filter_shape, stddev=0.1), name="W_conv")
            conv = tf.nn.conv2d(self.desc_batch_expanded,
                                W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="desc_conv")

            conv = tf.nn.relu(conv, name="conv_relu")
            conv = tf.nn.dropout(conv, keep_prob=self.dropout_keep_prob)

            # [B, (doclength-F+1), 1, K]
            # [B,K] - Global Average Pooling
            self.desc_encoded = tf.reduce_mean(conv, reduction_indices=[1,2])

            # [B, 1, K]
            self.desc_encoded_expand = tf.expand_dims(
                input=self.desc_encoded, dim=1)

            # [B, N]
            self.desc_scores = tf.reduce_sum(tf.mul(
              self.allentity_embs, self.desc_encoded_expand), 2)

            self.desc_posteriors = tf.nn.softmax(self.desc_scores,
                                                 name="entity_post_softmax")

    ###########   end def __init__      ##########################################

    def loss_graph(self, true_entity_ids, scope_name, device_gpu):
        """
        Compute the loss loss.

        Args:
            self: (todo): write your description
            true_entity_ids: (str): write your description
            scope_name: (str): write your description
            device_gpu: (todo): write your description
        """

        with tf.variable_scope(scope_name) as s, tf.device(device_gpu) as d:
            self.crossentropy_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=self.desc_scores,
              labels=true_entity_ids,
              name="desc_posterior_loss")

            self.wikiDescLoss = tf.reduce_sum(
                self.crossentropy_losses) / tf.to_float(self.batch_size)


            '''
            # Maximize cosine distance between true_entity_embeddings and encoded_description
            # max CosineDis(self.trueentity_embs, self.desc_encoded)

            # [B, 1] - NOT [B] Due to keep_dims
            trueen_emb_norm = tf.sqrt(
              tf.reduce_sum(tf.square(self.trueentity_embs), 1, keep_dims=True))
            # [B, K]
            true_emb_normalized = self.trueentity_embs / trueen_emb_norm

            # [B, 1] - NOT [B] Due to keep_dims
            negen_emb_norm = tf.sqrt(
              tf.reduce_sum(tf.square(self.negentity_embs), 1, keep_dims=True))
            # [B, K]
            neg_emb_normalized = self.negentity_embs / negen_emb_norm


            # [B, 1] - NOT [B] Due to keep_dims
            desc_enc_norm = tf.sqrt(
              tf.reduce_sum(tf.square(self.desc_encoded), 1, keep_dims=True))
            # [B, K]
            desc_end_normalized = self.desc_encoded / desc_enc_norm

            # [B]
            self.true_cosDist = tf.reduce_mean(tf.mul(true_emb_normalized, desc_end_normalized))

            self.neg_cosDist = tf.reduce_mean(tf.mul(neg_emb_normalized, desc_end_normalized))

            # Loss = -ve dot_prod because distance has to be decreased
            #self.wikiDescLoss = self.neg_cosDist - self.true_cosDist
            '''
