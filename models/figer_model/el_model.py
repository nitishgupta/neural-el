import time
import tensorflow as tf
import numpy as np
import random
import sys
import gc

from evaluation import evaluate_inference
from evaluation import evaluate_types, evaluate_el
from models.base import Model
from models.figer_model.context_encoder import ContextEncoderModel
from models.figer_model.coherence_model import CoherenceModel
from models.figer_model.wiki_desc import WikiDescModel
from models.figer_model.labeling_model import LabelingModel
from models.figer_model.entity_posterior import EntityPosterior
from models.figer_model.loss_optim import LossOptim


np.set_printoptions(precision=5)


class ELModel(Model):
    """Unsupervised Clustering using Discrete-State VAE"""

    def __init__(self, sess, reader, dataset, max_steps, pretrain_max_steps,
                 word_embed_dim, context_encoded_dim,
                 context_encoder_lstmsize, context_encoder_num_layers,
                 coherence_numlayers, jointff_numlayers,
                 learning_rate, dropout_keep_prob, reg_constant,
                 checkpoint_dir, optimizer, mode='train', strict=False,
                 pretrain_word_embed=True, typing=True, el=True,
                 coherence=False, textcontext=False, useCNN=False,
                 WDLength=100, Fsize=5, entyping=True):
        self.optimizer = optimizer
        self.mode = mode
        self.sess = sess
        self.reader = reader  # Reader class
        self.dataset = dataset
        self.strict = strict
        self.pretrain_word_embed = pretrain_word_embed
        assert self.pretrain_word_embed, "Only use pretrained word embeddings"
        self.typing = typing    # Bool - Perform typing
        self.el = el    # Bool - Perform Entity-Linking
        self.coherence = coherence
        self.textcontext = textcontext
        if not (self.coherence or self.textcontext):
            print("Both textcontext and coherence cannot be False")
            sys.exit(0)
        self.useCNN = useCNN
        self.WDLength = WDLength
        self.Fsize = Fsize
        self.entyping = entyping


        self.max_steps = max_steps  # Max num of steps of training to run
        self.pretrain_max_steps = pretrain_max_steps
        self.batch_size = reader.batch_size
        self.reg_constant = reg_constant
        self.dropout_keep_prob = dropout_keep_prob
        self.lr = learning_rate

        # Num of clusters = Number of entities in dataset.
        self.num_labels = self.reader.num_labels
        self.num_words = self.reader.num_words
        self.num_knwn_entities = self.reader.num_knwn_entities
        self.num_cand_entities = self.reader.num_cands

        # Size of word embeddings
        if not self.pretrain_word_embed:
            self.word_embed_dim = word_embed_dim
        else:
            self.word_embed_dim = 300

        # Context encoders
        self.context_encoded_dim = context_encoded_dim
        self.context_encoder_lstmsize = context_encoder_lstmsize
        self.context_encoder_num_layers = context_encoder_num_layers
        # Coherence Encoder
        self.coherence_numlayers = coherence_numlayers
        # Joint FeedForward
        self.jointff_numlayers = jointff_numlayers

        self.checkpoint_dir = checkpoint_dir

        self.embeddings_scope = "embeddings"
        self.word_embed_var_name = "word_embeddings"
        self.encoder_model_scope = "context_encoder"
        self.coherence_model_scope = "coherence_encoder"
        self.wikidesc_model_scope = "wikidesc_encoder"
        self.joint_context_scope = "joint_context"
        self.label_model_scope = "labeling_model"
        self.labeling_loss_scope = "labeling_loss"
        self.entity_posterior_scope = "en_posterior_model"
        self.posterior_loss_scope = "en_posterior_loss"
        self.wikidesc_loss_scope = "wikidesc_loss"
        self.optim_scope = "labeling_optimization"

        self._attrs=[
          "textcontext", "coherence", "typing",
          "pretrain_word_embed", "word_embed_dim", "num_words", "num_labels",
          "num_knwn_entities", "context_encoded_dim", "context_encoder_lstmsize",
          "context_encoder_num_layers", "coherence_numlayers",
          "reg_constant", "strict", "lr", "optimizer"]


        #GPU Allocations
        self.device_placements = {
          'cpu': '/cpu:0',
          'gpu': '/gpu:0'
        }

        with tf.variable_scope("figer_model") as scope:
            self.learning_rate = tf.Variable(self.lr, name='learning_rate',
                                             trainable=False)
            self.global_step = tf.Variable(0, name='global_step', trainable=False,
                                           dtype=tf.int32)
            self.increment_global_step_op = self.global_step.assign_add(1)

            self.build_placeholders()

            # Encoder Models : Name LSTM, Text FF and Links FF networks
            with tf.variable_scope(self.encoder_model_scope) as scope:
                if self.pretrain_word_embed == False:
                    self.left_context_embeddings = tf.nn.embedding_lookup(
                      self.word_embeddings, self.left_batch, name="left_embeddings")
                    self.right_context_embeddings = tf.nn.embedding_lookup(
                      self.word_embeddings, self.right_batch, name="right_embeddings")


                if self.textcontext:
                    self.context_encoder_model = ContextEncoderModel(
                      num_layers=self.context_encoder_num_layers,
                      batch_size=self.batch_size,
                      lstm_size=self.context_encoder_lstmsize,
                      left_embed_batch=self.left_context_embeddings,
                      left_lengths=self.left_lengths,
                      right_embed_batch=self.right_context_embeddings,
                      right_lengths=self.right_lengths,
                      context_encoded_dim=self.context_encoded_dim,
                      scope_name=self.encoder_model_scope,
                      device=self.device_placements['gpu'],
                      dropout_keep_prob=self.dropout_keep_prob)

                if self.coherence:
                    self.coherence_model = CoherenceModel(
                      num_layers=self.coherence_numlayers,
                      batch_size=self.batch_size,
                      input_size=self.reader.num_cohstr,
                      coherence_indices=self.coherence_indices,
                      coherence_values=self.coherence_values,
                      coherence_matshape=self.coherence_matshape,
                      context_encoded_dim=self.context_encoded_dim,
                      scope_name=self.coherence_model_scope,
                      device=self.device_placements['gpu'],
                      dropout_keep_prob=self.dropout_keep_prob)

                if self.coherence and self.textcontext:
                    # [B, 2*context_encoded_dim]
                    joint_context_encoded = tf.concat(
                      1, [self.context_encoder_model.context_encoded,
                          self.coherence_model.coherence_encoded],
                      name='joint_context_encoded')

                    context_vec_size = 2*self.context_encoded_dim

                    ### WITH FF AFTER CONCAT  ##########
                    trans_weights = tf.get_variable(
                      name="joint_trans_weights",
                      shape=[context_vec_size, context_vec_size],
                      initializer=tf.random_normal_initializer(
                        mean=0.0,
                        stddev=1.0/(100.0)))

                    # [B, context_encoded_dim]
                    joint_context_encoded = tf.matmul(joint_context_encoded, trans_weights)
                    self.joint_context_encoded = tf.nn.relu(joint_context_encoded)
                    ####################################

                elif self.textcontext:
                    self.joint_context_encoded = self.context_encoder_model.context_encoded
                    context_vec_size = self.context_encoded_dim
                elif self.coherence:
                    self.joint_context_encoded = self.coherence_model.coherence_encoded
                    context_vec_size = self.context_encoded_dim
                else:
                    print("ERROR:Atleast one of local or "
                          "document context needed.")
                    sys.exit(0)

                self.posterior_model = EntityPosterior(
                  batch_size=self.batch_size,
                  num_knwn_entities=self.num_knwn_entities,
                  context_encoded_dim=context_vec_size,
                  context_encoded=self.joint_context_encoded,
                  entity_ids=self.sampled_entity_ids,
                  scope_name=self.entity_posterior_scope,
                  device_embeds=self.device_placements['gpu'],
                  device_gpu=self.device_placements['gpu'])

                self.labeling_model = LabelingModel(
                  batch_size=self.batch_size,
                  num_labels=self.num_labels,
                  context_encoded_dim=context_vec_size,
                  true_entity_embeddings=self.posterior_model.trueentity_embeddings,
                  word_embed_dim=self.word_embed_dim,
                  context_encoded=self.joint_context_encoded,
                  mention_embed=None,
                  scope_name=self.label_model_scope,
                  device=self.device_placements['gpu'])

                if self.useCNN:
                    self.wikidescmodel = WikiDescModel(
                      desc_batch=self.wikidesc_batch,
                      trueentity_embs=self.posterior_model.trueentity_embeddings,
                      negentity_embs=self.posterior_model.negentity_embeddings,
                      allentity_embs=self.posterior_model.sampled_entity_embeddings,
                      batch_size=self.batch_size,
                      doclength=self.WDLength,
                      wordembeddim=self.word_embed_dim,
                      filtersize=self.Fsize,
                      desc_encoded_dim=context_vec_size,
                      scope_name=self.wikidesc_model_scope,
                      device=self.device_placements['gpu'],
                      dropout_keep_prob=self.dropout_keep_prob)
            #end - encoder variable scope


        # Encoder FF Variables + Cluster Embedding
        self.train_vars = tf.trainable_variables()

        self.loss_optim = LossOptim(self)
    ################ end Initialize  #############################################


    def build_placeholders(self):
        # Left Context
        self.left_batch = tf.placeholder(
          tf.int32, [self.batch_size, None], name="left_batch")
        self.left_context_embeddings = tf.placeholder(
          tf.float32, [self.batch_size, None, self.word_embed_dim], name="left_embeddings")
        self.left_lengths = tf.placeholder(
          tf.int32, [self.batch_size], name="left_lengths")

        # Right Context
        self.right_batch = tf.placeholder(
          tf.int32, [self.batch_size, None], name="right_batch")
        self.right_context_embeddings = tf.placeholder(
          tf.float32, [self.batch_size, None, self.word_embed_dim], name="right_embeddings")
        self.right_lengths = tf.placeholder(
          tf.int32, [self.batch_size], name="right_lengths")

        # Mention Embedding
        self.mention_embed = tf.placeholder(
          tf.float32, [self.batch_size, self.word_embed_dim], name="mentions_embed")

        # Wiki Description Batch
        self.wikidesc_batch = tf.placeholder(
          tf.float32, [self.batch_size, self.WDLength, self.word_embed_dim],
          name="wikidesc_batch")

        # Labels
        self.labels_batch = tf.placeholder(
          tf.float32, [self.batch_size, self.num_labels], name="true_labels")

        # Candidates, Priors and True Entities Ids
        self.sampled_entity_ids = tf.placeholder(
          tf.int32, [self.batch_size, self.num_cand_entities], name="sampled_candidate_entities")

        self.entity_priors = tf.placeholder(
          tf.float32, [self.batch_size, self.num_cand_entities], name="entitiy_priors")

        self.true_entity_ids = tf.placeholder(
          tf.int32, [self.batch_size], name="true_entities_in_sampled")

        # Coherence
        self.coherence_indices = tf.placeholder(
          tf.int64, [None, 2], name="coherence_indices")

        self.coherence_values = tf.placeholder(
          tf.float32, [None], name="coherence_values")

        self.coherence_matshape = tf.placeholder(
          tf.int64, [2], name="coherence_matshape")

        #END-Placeholders

        if self.pretrain_word_embed == False:
            with tf.variable_scope(self.embeddings_scope) as s:
                with tf.device(self.device_placements['cpu']) as d:
                    self.word_embeddings = tf.get_variable(
                      name=self.word_embed_var_name,
                      shape=[self.num_words, self.word_embed_dim],
                      initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=(1.0/100.0)))

    def training_setup(self):
        # Make the loss graph
        print("[#] Making Loss Graph ....")
        self.loss_optim.make_loss_graph()

        print("[#] Defining pretraining losses and optimizers ...")
        self.loss_optim.label_optimization(
          trainable_vars=self.train_vars,
          optim_scope=self.optim_scope)

        print("All Trainable Variables")
        self.print_variables_in_collection(tf.trainable_variables())

    def training(self):
        self.training_setup()
        vars_tostore = tf.all_variables()
        saver = tf.train.Saver(var_list=vars_tostore, max_to_keep=30)

        # (Try) Load all pretraining model variables
        # If graph not found - Initialize trainable + optim variables
        print("Loading pre-saved checkpoint...")
        load_status = self.load(saver=saver,
                                checkpoint_dir=self.checkpoint_dir,
                                attrs=self._attrs)
        if not load_status:
            print("No checkpoint found. Training from scratch")
            self.sess.run(tf.initialize_variables(vars_tostore))

        start_iter = self.global_step.eval()
        start_time = time.time()

        print("[#] Pre-Training iterations done: %d" % start_iter)

        data_loading = 0
        tf.get_default_graph().finalize()

        for iteration in range(start_iter, self.max_steps):
            dstime = time.time()
            # GET BATCH
            (left_batch, left_lengths,
             right_batch, right_lengths,
             wikidesc_batch,
             labels_batch, coherence_batch,
             wid_idxs_batch, wid_cprobs_batch) = self.reader.next_train_batch()
            (coh_indices, coh_values, coh_matshape) = coherence_batch

            dtime = time.time() - dstime
            data_loading += dtime

            # FEED DICT
            feed_dict = {self.wikidesc_batch: wikidesc_batch,
                         self.sampled_entity_ids: wid_idxs_batch,
                         self.true_entity_ids: [0]*self.batch_size,
                         self.entity_priors: wid_cprobs_batch}

            if self.typing or self.entyping:
                type_dict = {self.labels_batch: labels_batch}
                feed_dict.update(type_dict)

            if self.textcontext:
                if not self.pretrain_word_embed:
                    context_dict = {
                      self.left_batch: left_batch,
                      self.right_batch: right_batch,
                      self.left_lengths: left_lengths,
                      self.right_lengths: right_lengths}
                    feed_dict.update(context_dict)
                else:
                    context_dict = {
                      self.left_context_embeddings: left_batch,
                      self.right_context_embeddings: right_batch,
                      self.left_lengths: left_lengths,
                      self.right_lengths: right_lengths}
                    feed_dict.update(context_dict)
            if self.coherence:
                coherence_dict = {self.coherence_indices: coherence_batch[0],
                                  self.coherence_values: coherence_batch[1],
                                  self.coherence_matshape: coherence_batch[2]}
                feed_dict.update(coherence_dict)

            # FETCH TENSORS
            fetch_tensors = [self.loss_optim.labeling_loss,
                             self.labeling_model.label_probs,
                             self.loss_optim.posterior_loss,
                             self.posterior_model.entity_posteriors,
                             self.loss_optim.entity_labeling_loss]
            if self.useCNN:
                fetch_tensors.append(self.loss_optim.wikidesc_loss)

            (fetches_old,
             _,
             _) = self.sess.run([fetch_tensors,
                                 self.loss_optim.optim_op,
                                 self.increment_global_step_op],
                                feed_dict=feed_dict)
            [old_loss, old_label_sigms,
             old_post_loss, old_posts,
             enLabelLoss] = fetches_old[0:5]
            if self.useCNN:
                [oldCNNLoss, trueCosDis,
                 negCosDis] = [fetches_old[5], 0.0, 0.0]
            else:
                [oldCNNLoss, trueCosDis, negCosDis] = [0.0, 0.0, 0.0]
            '''
            fetches_new = self.sess.run(fetch_tensors,
                                        feed_dict=feed_dict)
            [new_loss, new_label_sigms,
             new_post_loss,  new_posts] = fetches_new
            '''

            if iteration % 100 == 0:
                # [B, L]
                old_corr_preds, old_precision = evaluate.strict_pred(
                  labels_batch, old_label_sigms)
                context_preds = evaluate.correct_context_prediction(
                  old_posts, self.batch_size)

                print("Iter %2d, Epoch %d, T %4.2f secs, "
                      "Labeling Loss %.3f, EnLabelLoss %.3f"
                      % (iteration, self.reader.tr_epochs, time.time() - start_time,
                         old_loss, enLabelLoss))
                print("Old Posterior Loss : {0:.3f}, CNN Loss: {1:.3f} "
                      "TrueCos: {2:.3f} NegCos: {3:.3f}".format(
                      old_post_loss, oldCNNLoss, trueCosDis, negCosDis))
                print("[OLD] Num of strict correct predictions : {}, {}".format(
                      old_corr_preds, old_precision))
                print("[OLD] Num of correct context predictions : {}".format(
                    context_preds))

                print("Time to load data : %4.2f seconds \n" % data_loading)
                data_loading = 0

            if iteration != 0 and iteration % 500 == 0:
                self.save(saver=saver,
                          checkpoint_dir=self.checkpoint_dir,
                          attrs=self._attrs,
                          global_step=self.global_step)
                self.validation_performance(data_type=1, verbose=False)
                self.validation_performance(data_type=2, verbose=False)

            if iteration % 5000 == 0:
                print("Collecting garbage.")
                gc.collect()
    #end training

    # #####################      TEST     ##################################
    def load_ckpt_model(self, ckptpath=None):
        saver = tf.train.Saver(var_list=tf.all_variables())
        # (Try) Load all pretraining model variables
        print("Loading pre-saved model...")
        load_status = self.loadCKPTPath(saver=saver, ckptPath=ckptpath)

        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        tf.get_default_graph().finalize()


    def inference(self, ckptpath=None):
        saver = tf.train.Saver(var_list=tf.all_variables())
        # (Try) Load all pretraining model variables
        print("Loading pre-saved model...")
        load_status = self.loadCKPTPath(saver=saver, ckptPath=ckptpath)

        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        tf.get_default_graph().finalize()

        r = self.inference_run()
        return r

    def inference_run(self):
        assert self.reader.typeOfReader == "inference"
        # assert self.reader.batch_size == 1
        self.reader.reset_test()
        numInstances = 0

        # For types: List contains numpy matrices with row_size = BatchSize
        predLabelScoresnumpymat_list = []
        # For EL : Lists contain one list per mention
        widIdxs_list = []       # Candidate WID IDXs (First is true)
        condProbs_list = []     # Crosswikis conditional priors
        contextProbs_list = []  # Predicted Entity prob using context

        while self.reader.epochs < 1:
            (left_batch, left_lengths,
             right_batch, right_lengths,
             coherence_batch,
             wid_idxs_batch, wid_cprobs_batch) = self.reader.next_test_batch()

            # Candidates for entity linking
            # feed_dict = {self.sampled_entity_ids: wid_idxs_batch,
            #              self.entity_priors: wid_cprobs_batch}
            feed_dict = {self.sampled_entity_ids: wid_idxs_batch}
            # Required Context
            if self.textcontext:
                if not self.pretrain_word_embed:
                    context_dict = {
                      self.left_batch: left_batch,
                      self.right_batch: right_batch,
                      self.left_lengths: left_lengths,
                      self.right_lengths: right_lengths}
                else:
                    context_dict = {
                      self.left_context_embeddings: left_batch,
                      self.right_context_embeddings: right_batch,
                      self.left_lengths: left_lengths,
                      self.right_lengths: right_lengths}
                feed_dict.update(context_dict)
            if self.coherence:
                coherence_dict = {self.coherence_indices: coherence_batch[0],
                                  self.coherence_values: coherence_batch[1],
                                  self.coherence_matshape: coherence_batch[2]}
                feed_dict.update(coherence_dict)

            fetch_tensors = [self.labeling_model.label_probs,
                             self.posterior_model.entity_posteriors]

            fetches = self.sess.run(fetch_tensors, feed_dict=feed_dict)

            [label_sigms, context_probs] = fetches

            predLabelScoresnumpymat_list.append(label_sigms)
            condProbs_list.extend(wid_cprobs_batch)
            widIdxs_list.extend(wid_idxs_batch)
            contextProbs_list.extend(context_probs.tolist())
            numInstances += self.reader.batch_size

        # print("Num of instances {}".format(numInstances))
        # print("Starting Type and EL Evaluations ... ")
        # pred_TypeSetsList: [B, Types], For each mention, list of pred types
        pred_TypeSetsList = evaluate_types.evaluate(
            predLabelScoresnumpymat_list,
            self.reader.idx2label)

        # evWTs:For each mention: Contains a list of [WTs, WIDs, Probs]
        # Each element above has (MaxPrior, MaxContext, MaxJoint)
        # sortedContextWTs: Titles sorted in decreasing context prob
        (jointProbs_list,
         evWTs,
         sortedContextWTs) = evaluate_inference.evaluateEL(
            condProbs_list, widIdxs_list, contextProbs_list,
            self.reader.idx2knwid, self.reader.wid2WikiTitle,
            verbose=False)

        return (predLabelScoresnumpymat_list,
                widIdxs_list, condProbs_list, contextProbs_list,
                jointProbs_list, evWTs, pred_TypeSetsList)


    def dataset_test(self, ckptpath=None):
        saver = tf.train.Saver(var_list=tf.all_variables())
        # (Try) Load all pretraining model variables
        print("Loading pre-saved model...")
        load_status = self.loadCKPTPath(saver=saver, ckptPath=ckptpath)

        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)
        tf.get_default_graph().finalize()

        returns = self.dataset_performance()
        return returns

    def dataset_performance(self):
        print("Test accuracy starting ... ")
        assert self.reader.typeOfReader=="test"
        # assert self.reader.batch_size == 1
        self.reader.reset_test()
        numInstances = 0

        stime = time.time()

        # For types: List contains numpy matrices of row_size = BatchSize
        trueLabelScoresnumpymat_list = []
        predLabelScoresnumpymat_list = []
        # For EL : Lists contain one list per mention
        condProbs_list = []    # Crosswikis conditional priors
        widIdxs_list = []      # Candidate WID IDXs (First is true)
        contextProbs_list = [] # Predicted Entity prob using context

        while self.reader.epochs < 1:
            (left_batch, left_lengths,
             right_batch, right_lengths,
             # wikidesc_batch,
             labels_batch, coherence_batch,
             wid_idxs_batch, wid_cprobs_batch) = self.reader.next_test_batch()

            # Candidates for entity linking
            feed_dict = {#self.wikidesc_batch: wikidesc_batch,
                         self.sampled_entity_ids: wid_idxs_batch,
                         self.entity_priors: wid_cprobs_batch}
            # Required Context
            if self.textcontext:
                if self.pretrain_word_embed == False:
                    context_dict = {
                      self.left_batch: left_batch, self.right_batch: right_batch,
                      self.left_lengths: left_lengths, self.right_lengths: right_lengths}
                else:
                    context_dict = {
                      self.left_context_embeddings: left_batch, self.right_context_embeddings: right_batch,
                      self.left_lengths: left_lengths, self.right_lengths: right_lengths}
                feed_dict.update(context_dict)
            if self.coherence:
                coherence_dict = {self.coherence_indices: coherence_batch[0],
                                  self.coherence_values: coherence_batch[1],
                                  self.coherence_matshape: coherence_batch[2]}
                feed_dict.update(coherence_dict)

            fetch_tensors = [self.labeling_model.label_probs,
                             self.posterior_model.entity_posteriors]

            fetches = self.sess.run(fetch_tensors, feed_dict=feed_dict)

            [label_sigms, context_probs] = fetches

            trueLabelScoresnumpymat_list.append(labels_batch)
            predLabelScoresnumpymat_list.append(label_sigms)
            condProbs_list.extend(wid_cprobs_batch)
            widIdxs_list.extend(wid_idxs_batch)
            contextProbs_list.extend(context_probs.tolist())
            numInstances += self.reader.batch_size

        print("Num of instances {}".format(numInstances))
        print("Starting Type and EL Evaluations ... ")
        # evaluate_types.evaluate(
        #   trueLabelScoresnumpymat_list, predLabelScoresnumpymat_list,
        #   condProbs_list, widIdxs_list, contextProbs_list,
        #   self.reader.wid2WikiTitle, self.reader.wid2TypeLabels,
        #   self.reader.idx2label)
        # evaluate.types_predictions(
        #   trueLabelScoresnumpymat_list, predLabelScoresnumpymat_list)
        (jointProbs_list,
         evWTs,
         sortedContextWTs) = evaluate_el.evaluateEL(
            condProbs_list, widIdxs_list, contextProbs_list,
            self.reader.idx2knwid, self.reader.wid2WikiTitle,
            verbose=False)

        return (widIdxs_list, condProbs_list, contextProbs_list,
                jointProbs_list, evWTs, sortedContextWTs)

    def softmax(self, scores):
        expc = np.exp(scores)
        sumc = np.sum(expc)
        softmax_out = expc/sumc
        return softmax_out

    def print_all_variables(self):
        print("All Variables in the graph : ")
        self.print_variables_in_collection(tf.all_variables())

    def print_trainable_variables(self):
        print("All Trainable variables in the graph : ")
        self.print_variables_in_collection(tf.trainable_variables())

    def print_variables_in_collection(self, list_vars):
        print("Variables in list: ")
        for var in list_vars:
            print("  %s" % var.name)

    def extractEntityEmbeddings(self, ckptPath=None):
        saver = tf.train.Saver(var_list=tf.all_variables())
        print("Loading pre-saved model...")
        load_status = self.loadCKPTPath(saver=saver, ckptPath=ckptPath)
        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)
        tf.get_default_graph().finalize()

        enembs = self.sess.run(self.posterior_model.knwn_entity_embeddings)
        return enembs
