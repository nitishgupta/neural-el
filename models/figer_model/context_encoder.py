import time
import tensorflow as tf
import numpy as np

from models.base import Model

class ContextEncoderModel(Model):
    """Run Forward and Backward LSTM and concatenate last outputs to get
       context representation"""

    def __init__(self, num_layers, batch_size, lstm_size,
                 left_embed_batch, left_lengths, right_embed_batch, right_lengths,
                 context_encoded_dim, scope_name, device, dropout_keep_prob=1.0):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            num_layers: (int): write your description
            batch_size: (int): write your description
            lstm_size: (int): write your description
            left_embed_batch: (todo): write your description
            left_lengths: (int): write your description
            right_embed_batch: (int): write your description
            right_lengths: (int): write your description
            context_encoded_dim: (str): write your description
            scope_name: (str): write your description
            device: (todo): write your description
            dropout_keep_prob: (str): write your description
        """

        self.num_layers = num_layers  # Num of layers in the encoder and decoder network
        self.num_lstm_layers = 1

        # Left / Right Context Dim.
        # Context Representation Dim : 2*lstm_size
        self.lstm_size = lstm_size
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_size = batch_size
        self.context_encoded_dim = context_encoded_dim
        self.left_context_embeddings = left_embed_batch
        self.right_context_embeddings = right_embed_batch

        with tf.variable_scope(scope_name) as sc, tf.device(device) as d:
            with tf.variable_scope("left_encoder") as s:
                l_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
                    self.lstm_size, state_is_tuple=True)

                l_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(
                  cell=l_encoder_cell,
                  input_keep_prob=self.dropout_keep_prob,
                  output_keep_prob=self.dropout_keep_prob)

                self.left_encoder = tf.nn.rnn_cell.MultiRNNCell(
                  [l_dropout_cell] * self.num_lstm_layers, state_is_tuple=True)

                self.left_outputs, self.left_states = tf.nn.dynamic_rnn(
                  cell=self.left_encoder, inputs=self.left_context_embeddings,
                  sequence_length=left_lengths, dtype=tf.float32)

            with tf.variable_scope("right_encoder") as s:
                r_encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
                    self.lstm_size, state_is_tuple=True)

                r_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(
                  cell=r_encoder_cell,
                  input_keep_prob=self.dropout_keep_prob,
                  output_keep_prob=self.dropout_keep_prob)

                self.right_encoder = tf.nn.rnn_cell.MultiRNNCell(
                  [r_dropout_cell] * self.num_lstm_layers, state_is_tuple=True)

                self.right_outputs, self.right_states = tf.nn.dynamic_rnn(
                  cell=self.right_encoder,
                  inputs=self.right_context_embeddings,
                  sequence_length=right_lengths, dtype=tf.float32)

            # Left Context Encoded
            # [B, LSTM_DIM]
            self.left_last_output = self.get_last_output(
              outputs=self.left_outputs, lengths=left_lengths,
              name="left_context_encoded")

            # Right Context Encoded
            # [B, LSTM_DIM]
            self.right_last_output = self.get_last_output(
              outputs=self.right_outputs, lengths=right_lengths,
              name="right_context_encoded")

            # Context Encoded Vector
            self.context_lstm_encoded = tf.concat(
              1, [self.left_last_output, self.right_last_output],
              name='context_lstm_encoded')

            # Linear Transformation to get context_encoded_dim
            # Layer 1
            self.trans_weights = tf.get_variable(
              name="context_trans_weights",
              shape=[2*self.lstm_size, self.context_encoded_dim],
              initializer=tf.random_normal_initializer(
                mean=0.0,
                stddev=1.0/(100.0)))

            # [B, context_encoded_dim]
            context_encoded = tf.matmul(self.context_lstm_encoded,
                                        self.trans_weights)
            context_encoded = tf.nn.relu(context_encoded)

            self.hidden_layers = []
            for i in range(1, self.num_layers):
                weight_matrix = tf.get_variable(
                  name="context_hlayer_"+str(i),
                  shape=[self.context_encoded_dim, self.context_encoded_dim],
                  initializer=tf.random_normal_initializer(
                    mean=0.0,
                    stddev=1.0/(100.0)))
                self.hidden_layers.append(weight_matrix)

            for i in range(1, self.num_layers):
                context_encoded = tf.nn.dropout(
                    context_encoded, keep_prob=self.dropout_keep_prob)
                context_encoded = tf.matmul(context_encoded,
                                            self.hidden_layers[i-1])
                context_encoded = tf.nn.relu(context_encoded)

            self.context_encoded = tf.nn.dropout(
                context_encoded, keep_prob=self.dropout_keep_prob)

    def get_last_output(self, outputs, lengths, name):
        """
        Returns the last batch of sequences.

        Args:
            self: (todo): write your description
            outputs: (todo): write your description
            lengths: (int): write your description
            name: (str): write your description
        """
        reverse_output = tf.reverse_sequence(input=outputs,
                                             seq_lengths=tf.to_int64(lengths),
                                             seq_dim=1,
                                             batch_dim=0)
        en_last_output = tf.slice(input_=reverse_output,
                                  begin=[0,0,0],
                                  size=[self.batch_size, 1, -1])
        # [batch_size, h_dim]
        encoder_last_output = tf.reshape(en_last_output,
                                         shape=[self.batch_size, -1],
                                         name=name)

        return encoder_last_output
