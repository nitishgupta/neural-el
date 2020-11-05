import os
import sys
import tensorflow as tf
import numpy as np

import readers.utils as utils
from evaluation import evaluate
from evaluation import evaluate_el
from evaluation import evaluate_types
from models.base import Model
from models.figer_model.context_encoder import ContextEncoderModel
from models.figer_model.coherence_model import CoherenceModel
from models.figer_model.wiki_desc import WikiDescModel
from models.figer_model.joint_context import JointContextModel
from models.figer_model.labeling_model import LabelingModel
from models.figer_model.entity_posterior import EntityPosterior
from models.figer_model.loss_optim import LossOptim


class ColdStart(object):
    def __init__(self, figermodel):
        """
        Initialize the network

        Args:
            self: (todo): write your description
            figermodel: (todo): write your description
        """
        print("######   ENTERED THE COLD WORLD OF THE UNKNOWN    ##############")
        # Object of the WikiELModel Class
        self.fm = figermodel
        self.coldDir = self.fm.reader.coldDir
        coldWid2DescVecs_pkl = os.path.join(self.coldDir, "coldwid2descvecs.pkl")
        self.coldWid2DescVecs = utils.load(coldWid2DescVecs_pkl)
        self.num_cold_entities = self.fm.reader.num_cold_entities
        self.batch_size = self.fm.batch_size
        (self.coldwid2idx,
         self.idx2coldwid) = (self.fm.reader.coldwid2idx, self.fm.reader.idx2coldwid)

    def _makeDescLossGraph(self):
        """
        Creates the graph for the model.

        Args:
            self: (todo): write your description
        """
        with tf.variable_scope("cold") as s:
            with tf.device(self.fm.device_placements['gpu']) as d:
                tf.set_random_seed(1)

                self.coldEnEmbsToAssign = tf.placeholder(
                  tf.float32, [self.num_cold_entities, 200], name="coldEmbsAssignment")

                self.coldEnEmbs = tf.get_variable(
                  name="cold_entity_embeddings",
                  shape=[self.num_cold_entities, 200],
                  initializer=tf.random_normal_initializer(mean=-0.25,
                                                           stddev=1.0/(100.0)))

                self.assignColdEmbs = self.coldEnEmbs.assign(self.coldEnEmbsToAssign)

                self.trueColdEnIds = tf.placeholder(
                  tf.int32, [self.batch_size], name="true_entities_idxs")

                # Should be a list of zeros
                self.softTrueIdxs = tf.placeholder(
                  tf.int32, [self.batch_size], name="softmaxTrueEnsIdxs")

                # [B, D]
                self.trueColdEmb = tf.nn.embedding_lookup(
                    self.coldEnEmbs, self.trueColdEnIds)
                # [B, 1, D]
                self.trueColdEmb_exp = tf.expand_dims(
                  input=self.trueColdEmb, dim=1)

                self.label_scores = tf.matmul(self.trueColdEmb,
                                              self.fm.labeling_model.label_weights)

                self.labeling_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=self.label_scores,
                  targets=self.fm.labels_batch,
                  name="labeling_loss")

                self.labelingLoss = tf.reduce_sum(
                  self.labeling_losses) / tf.to_float(self.batch_size)

                # [B, D]
                self.descEncoded = self.fm.wikidescmodel.desc_encoded

                ## Maximize sigmoid of dot-prod between true emb. and desc encoding
                descLosses = -tf.sigmoid(tf.reduce_sum(tf.mul(self.trueColdEmb, self.descEncoded), 1))
                self.descLoss = tf.reduce_sum(descLosses)/tf.to_float(self.batch_size)


                # L-2 Norm Loss
                self.trueEmbNormLoss = tf.reduce_sum(
                  tf.square(self.trueColdEmb))/(tf.to_float(self.batch_size))


                ''' Concat trueColdEmb_exp to negKnownEmbs so that 0 is the true entity.
                Dotprod this emb matrix with descEncoded to get scores and apply softmax
                '''

                self.trcoldvars = self.fm.scope_vars_list(scope_name="cold",
                                                          var_list=tf.trainable_variables())

                print("Vars in Training")
                for var in self.trcoldvars:
                    print(var.name)


                self.optimizer = tf.train.AdamOptimizer(
                  learning_rate=self.fm.learning_rate,
                  name='AdamCold_')

                self.total_labeling_loss = self.labelingLoss + self.trueEmbNormLoss
                self.label_gvs = self.optimizer.compute_gradients(
                  loss=self.total_labeling_loss, var_list=self.trcoldvars)
                self.labeling_optim_op = self.optimizer.apply_gradients(self.label_gvs)

                self.total_loss = self.labelingLoss + 100*self.descLoss + self.trueEmbNormLoss
                self.comb_gvs = self.optimizer.compute_gradients(
                  loss=self.total_loss, var_list=self.trcoldvars)
                self.combined_optim_op = self.optimizer.apply_gradients(self.comb_gvs)


                self.allcoldvars = self.fm.scope_vars_list(scope_name="cold",
                                                           var_list=tf.all_variables())

                print("All Vars in Cold")
                for var in self.allcoldvars:
                    print(var.name)

        print("Loaded and graph made")
    ###  GRAPH COMPLETE  ###

    #############################################################################
    def _trainColdEmbFromTypes(self, epochsToTrain=5):
        """
        Train the model.

        Args:
            self: (todo): write your description
            epochsToTrain: (todo): write your description
        """
        print("Training Cold Entity Embeddings from Typing Info")

        epochsDone = self.fm.reader.val_epochs

        while self.fm.reader.val_epochs < epochsToTrain:
            (left_batch, left_lengths,
             right_batch, right_lengths,
             wids_batch,
             labels_batch, coherence_batch,
             wid_idxs_batch, wid_cprobs_batch) = self.fm.reader._next_padded_batch(data_type=1)

            trueColdWidIdxsBatch = []
            trueColdWidDescWordVecBatch = []
            for wid in wids_batch:
                trueColdWidIdxsBatch.append(self.coldwid2idx[wid])
                trueColdWidDescWordVecBatch.append(self.coldWid2DescVecs[wid])

            feed_dict = {self.trueColdEnIds: trueColdWidIdxsBatch,
                         self.fm.labels_batch: labels_batch}

            fetch_tensor = [self.labelingLoss, self.trueEmbNormLoss]

            (fetches, _) = self.fm.sess.run([fetch_tensor,
                                             self.labeling_optim_op],
                                            feed_dict=feed_dict)

            labelingLoss = fetches[0]
            trueEmbNormLoss = fetches[1]

            print("LL : {}  NormLoss : {}".format(labelingLoss, trueEmbNormLoss))

            newedone = self.fm.reader.val_epochs
            if newedone > epochsDone:
                print("Epochs : {}".format(newedone))
                epochsDone = newedone

    #############################################################################
    def _trainColdEmbFromTypesAndDesc(self, epochsToTrain=5):
        """
        Training function.

        Args:
            self: (todo): write your description
            epochsToTrain: (todo): write your description
        """
        print("Training Cold Entity Embeddings from Typing Info")

        epochsDone = self.fm.reader.val_epochs

        while self.fm.reader.val_epochs < epochsToTrain:
            (left_batch, left_lengths,
             right_batch, right_lengths,
             wids_batch,
             labels_batch, coherence_batch,
             wid_idxs_batch, wid_cprobs_batch) = self.fm.reader._next_padded_batch(data_type=1)

            trueColdWidIdxsBatch = []
            trueColdWidDescWordVecBatch = []
            for wid in wids_batch:
                trueColdWidIdxsBatch.append(self.coldwid2idx[wid])
                trueColdWidDescWordVecBatch.append(self.coldWid2DescVecs[wid])

            feed_dict = {self.fm.wikidesc_batch: trueColdWidDescWordVecBatch,
                         self.trueColdEnIds: trueColdWidIdxsBatch,
                         self.fm.labels_batch: labels_batch}

            fetch_tensor = [self.labelingLoss, self.descLoss, self.trueEmbNormLoss]

            (fetches,_) = self.fm.sess.run([fetch_tensor,
                                            self.combined_optim_op],
                                           feed_dict=feed_dict)

            labelingLoss = fetches[0]
            descLoss = fetches[1]
            normLoss = fetches[2]

            print("L : {}  D : {}  NormLoss : {}".format(labelingLoss, descLoss, normLoss))

            newedone = self.fm.reader.val_epochs
            if newedone > epochsDone:
                print("Epochs : {}".format(newedone))
                epochsDone = newedone

    #############################################################################

    def runEval(self):
        """
        Run inference.

        Args:
            self: (todo): write your description
        """
        print("Running Evaluations")
        self.fm.reader.reset_validation()
        correct = 0
        total = 0
        totnew = 0
        correctnew = 0
        while self.fm.reader.val_epochs < 1:
            (left_batch, left_lengths,
             right_batch, right_lengths,
             wids_batch,
             labels_batch, coherence_batch,
             wid_idxs_batch,
             wid_cprobs_batch) = self.fm.reader._next_padded_batch(data_type=1)

            trueColdWidIdxsBatch = []

            for wid in wids_batch:
                trueColdWidIdxsBatch.append(self.coldwid2idx[wid])

            feed_dict = {self.fm.sampled_entity_ids: wid_idxs_batch,
                         self.fm.left_context_embeddings: left_batch,
                         self.fm.right_context_embeddings: right_batch,
                         self.fm.left_lengths: left_lengths,
                         self.fm.right_lengths: right_lengths,
                         self.fm.coherence_indices: coherence_batch[0],
                         self.fm.coherence_values: coherence_batch[1],
                         self.fm.coherence_matshape: coherence_batch[2],
                         self.trueColdEnIds: trueColdWidIdxsBatch}

            fetch_tensor = [self.trueColdEmb,
                            self.fm.joint_context_encoded,
                            self.fm.posterior_model.sampled_entity_embeddings,
                            self.fm.posterior_model.entity_scores]

            fetched_vals = self.fm.sess.run(fetch_tensor, feed_dict=feed_dict)
            [trueColdEmbs,            # [B, D]
             context_encoded,         # [B, D]
             neg_entity_embeddings,   # [B, N, D]
             neg_entity_scores] = fetched_vals    # [B, N]

            # [B]
            trueColdWidScores = np.sum(trueColdEmbs*context_encoded, axis=1)
            entity_scores = neg_entity_scores
            entity_scores[:,0] = trueColdWidScores
            context_entity_scores = np.exp(entity_scores)/np.sum(np.exp(entity_scores))

            maxIdxs = np.argmax(context_entity_scores, axis=1)
            for i in range(0, self.batch_size):
                total += 1
                if maxIdxs[i] == 0:
                    correct += 1

            scores_withpriors = context_entity_scores + wid_cprobs_batch

            maxIdxs = np.argmax(scores_withpriors, axis=1)
            for i in range(0, self.batch_size):
                totnew += 1
                if maxIdxs[i] == 0:
                    correctnew += 1

        print("Context T : {} C : {}".format(total, correct))
        print("WPriors T : {} C : {}".format(totnew, correctnew))

    ##############################################################################

    def typeBasedColdEmbExp(self, ckptName="FigerModel-20001"):
        ''' Train cold embeddings using wiki desc loss
        '''
        saver = tf.train.Saver(var_list=tf.all_variables())

        print("Loading Model ... ")
        if ckptName == None:
            print("Given CKPT Name")
            sys.exit()
        else:
            load_status = self.fm.loadSpecificCKPT(
              saver=saver, checkpoint_dir=self.fm.checkpoint_dir,
              ckptName=ckptName, attrs=self.fm._attrs)
        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        self._makeDescLossGraph()
        self.fm.sess.run(tf.initialize_variables(self.allcoldvars))
        self._trainColdEmbFromTypes(epochsToTrain=5)

        self.runEval()

    ##############################################################################

    def typeAndWikiDescBasedColdEmbExp(self, ckptName="FigerModel-20001"):
        ''' Train cold embeddings using wiki desc loss
        '''
        saver = tf.train.Saver(var_list=tf.all_variables())

        print("Loading Model ... ")
        if ckptName == None:
            print("Given CKPT Name")
            sys.exit()
        else:
            load_status = self.fm.loadSpecificCKPT(
              saver=saver, checkpoint_dir=self.fm.checkpoint_dir,
              ckptName=ckptName, attrs=self.fm._attrs)
        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        self._makeDescLossGraph()
        self.fm.sess.run(tf.initialize_variables(self.allcoldvars))
        self._trainColdEmbFromTypesAndDesc(epochsToTrain=5)

        self.runEval()

    # EVALUATION FOR COLD START WHEN INITIALIZING COLD EMB FROM WIKI DESC ENCODING
    def wikiDescColdEmbExp(self, ckptName="FigerModel-20001"):
        ''' Assign cold entity embeddings as wiki desc encoding
        '''
        assert self.batch_size == 1
        print("Loaded Cold Start Class. ")
        print("Size of cold entities : {}".format(len(self.coldWid2DescVecs)))

        saver = tf.train.Saver(var_list=tf.all_variables(), max_to_keep=5)

        print("Loading Model ... ")
        if ckptName == None:
            print("Given CKPT Name")
            sys.exit()
        else:
            load_status = self.fm.loadSpecificCKPT(
              saver=saver, checkpoint_dir=self.fm.checkpoint_dir,
              ckptName=ckptName, attrs=self.fm._attrs)
        if not load_status:
            print("No model to load. Exiting")
            sys.exit(0)

        iter_done = self.fm.global_step.eval()
        print("[#] Model loaded with iterations done: %d" % iter_done)

        self._makeDescLossGraph()
        self.fm.sess.run(tf.initialize_variables(self.allcoldvars))

        # Fill with encoded desc. in order of idx2coldwid
        print("Getting Encoded Description Vectors")
        descEncodedMatrix = []
        for idx in range(0, len(self.idx2coldwid)):
            wid = self.idx2coldwid[idx]
            desc_vec = self.coldWid2DescVecs[wid]
            feed_dict = {self.fm.wikidesc_batch: [desc_vec]}
            desc_encoded = self.fm.sess.run(self.fm.wikidescmodel.desc_encoded,
                                            feed_dict=feed_dict)
            descEncodedMatrix.append(desc_encoded[0])

        print("Initialization Experiment")
        self.runEval()

        print("Assigning Cold Embeddings from Wiki Desc Encoder ...")
        self.fm.sess.run(self.assignColdEmbs,
                         feed_dict={self.coldEnEmbsToAssign:descEncodedMatrix})

        print("After assigning based on Wiki Encoder")
        self.runEval()

    ##############################################################################
