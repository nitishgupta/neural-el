import time
import tensorflow as tf
import numpy as np

from models.base import Model

class LabelingModel(Model):
    """Unsupervised Clustering using Discrete-State VAE"""

    def __init__(self, batch_size, num_labels, context_encoded_dim,
                 true_entity_embeddings,
                 word_embed_dim, context_encoded, mention_embed, scope_name, device):
        """
        Initialize embeddings.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            num_labels: (int): write your description
            context_encoded_dim: (str): write your description
            true_entity_embeddings: (todo): write your description
            word_embed_dim: (int): write your description
            context_encoded: (str): write your description
            mention_embed: (str): write your description
            scope_name: (str): write your description
            device: (todo): write your description
        """

        self.batch_size = batch_size
        self.num_labels = num_labels
        self.word_embed_dim = word_embed_dim

        with tf.variable_scope(scope_name) as s, tf.device(device) as d:
            if mention_embed == None:
                self.label_weights = tf.get_variable(
                  name="label_weights",
                  shape=[context_encoded_dim, num_labels],
                  initializer=tf.random_normal_initializer(mean=0.0,
                                                           stddev=1.0/(100.0)))
            else:
                context_encoded = tf.concat(
                  1, [context_encoded, mention_embed], name='con_ment_repr')
                self.label_weights = tf.get_variable(
                  name="label_weights",
                  shape=[context_encoded_dim+word_embed_dim, num_labels],
                  initializer=tf.random_normal_initializer(mean=0.0,
                                                           stddev=1.0/(100.0)))

            # [B, L]
            self.label_scores = tf.matmul(context_encoded, self.label_weights)
            self.label_probs = tf.sigmoid(self.label_scores)

            ### PREDICT TYPES FROM ENTITIES
            #true_entity_embeddings = tf.nn.dropout(true_entity_embeddings, keep_prob=0.5)
            self.entity_label_scores = tf.matmul(true_entity_embeddings, self.label_weights)
            self.entity_label_probs = tf.sigmoid(self.label_scores)


    def loss_graph(self, true_label_ids, scope_name, device_gpu):
        """
        Perform a set of the model.

        Args:
            self: (todo): write your description
            true_label_ids: (str): write your description
            scope_name: (str): write your description
            device_gpu: (todo): write your description
        """
        with tf.variable_scope(scope_name) as s, tf.device(device_gpu) as d:
            # [B, L]
            self.cross_entropy_losses = tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.label_scores,
              targets=true_label_ids,
              name="labeling_loss")

            self.labeling_loss = tf.reduce_sum(
              self.cross_entropy_losses) / tf.to_float(self.batch_size)


            self.enlabel_cross_entropy_losses = tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.entity_label_scores,
              targets=true_label_ids,
              name="entity_labeling_loss")

            self.entity_labeling_loss = tf.reduce_sum(
              self.enlabel_cross_entropy_losses) / tf.to_float(self.batch_size)
