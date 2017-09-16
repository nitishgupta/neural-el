import time
import tensorflow as tf
import numpy as np

from models.base import Model

class JointContextModel(Model):
    """Entity Embeddings and Posterior Calculation"""

    def __init__(self, num_layers, context_encoded_dim, text_encoded,
                 coherence_encoded, scope_name, device, dropout_keep_prob):

        ''' Get context text and coherence encoded and combine into one repr.
        Input:
          text_encoded: Encoded vector for bi-LSTM. [context_encoded_dim]
          coherence_encoded: Encoded vector from sparse coherence FF [context_encoded_dim]

        Output:
          joint_encoded_vector: [context_encoded_dim]
        '''
        self.num_layers = num_layers
        self.dropout_keep_prob = dropout_keep_prob
        with tf.variable_scope(scope_name) as s:
            with tf.device(device) as d:
                self.joint_weights = tf.get_variable(
                  name="joint_context_layer",
                  shape=[2*context_encoded_dim, context_encoded_dim],
                  initializer=tf.random_normal_initializer(mean=0.0,
                                                           stddev=1.0/(100.0)))

                self.text_coh_concat = tf.concat(
                1, [text_encoded, coherence_encoded], name='text_coh_concat')

                context_encoded = tf.matmul(self.text_coh_concat, self.joint_weights)
                context_encoded = tf.nn.relu(context_encoded)

                self.hidden_layers = []
                for i in range(1, self.num_layers):
                    weight_matrix = tf.get_variable(
                      name="joint_context_hlayer_"+str(i),
                      shape=[context_encoded_dim, context_encoded_dim],
                      initializer=tf.random_normal_initializer(
                        mean=0.0,
                        stddev=1.0/(100.0)))
                    self.hidden_layers.append(weight_matrix)

                for i in range(1, self.num_layers):
                    context_encoded = tf.nn.dropout(context_encoded, keep_prob=self.dropout_keep_prob)
                    context_encoded = tf.matmul(context_encoded, self.hidden_layers[i-1])
                    context_encoded = tf.nn.relu(context_encoded)

                self.context_encoded = tf.nn.dropout(context_encoded, keep_prob=self.dropout_keep_prob)
