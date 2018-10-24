import os
import sys
import copy
import pprint
import numpy as np
import tensorflow as tf

from readers.inference_reader import InferenceReader
from readers.test_reader import TestDataReader
from models.figer_model.el_model import ELModel
from readers.config import Config
from readers.vocabloader import VocabLoader

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=7)

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("max_steps", 32000, "Maximum of iteration [450000]")
flags.DEFINE_integer("pretraining_steps", 32000, "Number of steps to run pretraining")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_string("model_path", "", "Path to trained model")
flags.DEFINE_string("dataset", "el-figer", "The name of dataset [ptb]")
flags.DEFINE_string("checkpoint_dir", "/tmp",
                    "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_integer("batch_size", 1, "Batch Size for training and testing")
flags.DEFINE_integer("word_embed_dim", 300, "Word Embedding Size")
flags.DEFINE_integer("context_encoded_dim", 100, "Context Encoded Dim")
flags.DEFINE_integer("context_encoder_num_layers", 1, "Num of Layers in context encoder network")
flags.DEFINE_integer("context_encoder_lstmsize", 100, "Size of context encoder hidden layer")
flags.DEFINE_integer("coherence_numlayers", 1, "Number of layers in the Coherence FF")
flags.DEFINE_integer("jointff_numlayers", 1, "Number of layers in the Coherence FF")
flags.DEFINE_integer("num_cand_entities", 30, "Num CrossWikis entity candidates")
flags.DEFINE_float("reg_constant", 0.00, "Regularization constant for NN weight regularization")
flags.DEFINE_float("dropout_keep_prob", 0.6, "Dropout Keep Probability")
flags.DEFINE_float("wordDropoutKeep", 0.6, "Word Dropout Keep Probability")
flags.DEFINE_float("cohDropoutKeep", 0.4, "Coherence Dropout Keep Probability")
flags.DEFINE_boolean("decoder_bool", True, "Decoder bool")
flags.DEFINE_string("mode", 'inference', "Mode to run")
flags.DEFINE_boolean("strict_context", False, "Strict Context exludes mention surface")
flags.DEFINE_boolean("pretrain_wordembed", True, "Use Word2Vec Embeddings")
flags.DEFINE_boolean("coherence", True, "Use Coherence")
flags.DEFINE_boolean("typing", True, "Perform joint typing")
flags.DEFINE_boolean("el", True, "Perform joint typing")
flags.DEFINE_boolean("textcontext", True, "Use text context from LSTM")
flags.DEFINE_boolean("useCNN", False, "Use wiki descp. CNN")
flags.DEFINE_boolean("glove", True, "Use Glove Embeddings")
flags.DEFINE_boolean("entyping", False, "Use Entity Type Prediction")
flags.DEFINE_integer("WDLength", 100, "Length of wiki description")
flags.DEFINE_integer("Fsize", 5, "For CNN filter size")

flags.DEFINE_string("optimizer", 'adam', "Optimizer to use. adagrad, adadelta or adam")

flags.DEFINE_string("config", 'configs/config.ini',
                    "VocabConfig Filepath")
flags.DEFINE_string("test_out_fp", "", "Write Test Prediction Data")

FLAGS = flags.FLAGS


def FLAGS_check(FLAGS):
    if not (FLAGS.textcontext and FLAGS.coherence):
        print("*** Local and Document context required ***")
        sys.exit(0)
    assert os.path.exists(FLAGS.model_path), "Model path doesn't exist."


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    FLAGS_check(FLAGS)

    config = Config(FLAGS.config, verbose=False)
    vocabloader = VocabLoader(config)
    
    if FLAGS.mode == 'inference':
        FLAGS.dropout_keep_prob = 1.0
        FLAGS.wordDropoutKeep = 1.0
        FLAGS.cohDropoutKeep = 1.0

        reader = InferenceReader(config=config,
                                 vocabloader=vocabloader,
                                 test_mens_file=config.test_file,
                                 num_cands=FLAGS.num_cand_entities,
                                 batch_size=FLAGS.batch_size,
                                 strict_context=FLAGS.strict_context,
                                 pretrain_wordembed=FLAGS.pretrain_wordembed,
                                 coherence=FLAGS.coherence)
        docta = reader.ccgdoc
        model_mode = 'inference'

    elif FLAGS.mode == 'test':
        FLAGS.dropout_keep_prob = 1.0
        FLAGS.wordDropoutKeep = 1.0
        FLAGS.cohDropoutKeep = 1.0

        reader = TestDataReader(config=config,
                                vocabloader=vocabloader,
                                test_mens_file=config.test_file,
                                num_cands=30,
                                batch_size=FLAGS.batch_size,
                                strict_context=FLAGS.strict_context,
                                pretrain_wordembed=FLAGS.pretrain_wordembed,
                                coherence=FLAGS.coherence)
        model_mode = 'test'

    else:
        print("MODE in FLAGS is incorrect : {}".format(FLAGS.mode))
        sys.exit()

    config_proto = tf.ConfigProto()
    config_proto.allow_soft_placement = True
    config_proto.gpu_options.allow_growth=True
    sess = tf.Session(config=config_proto)


    with sess.as_default():
        model = ELModel(
            sess=sess, reader=reader, dataset=FLAGS.dataset,
            max_steps=FLAGS.max_steps,
            pretrain_max_steps=FLAGS.pretraining_steps,
            word_embed_dim=FLAGS.word_embed_dim,
            context_encoded_dim=FLAGS.context_encoded_dim,
            context_encoder_num_layers=FLAGS.context_encoder_num_layers,
            context_encoder_lstmsize=FLAGS.context_encoder_lstmsize,
            coherence_numlayers=FLAGS.coherence_numlayers,
            jointff_numlayers=FLAGS.jointff_numlayers,
            learning_rate=FLAGS.learning_rate,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            reg_constant=FLAGS.reg_constant,
            checkpoint_dir=FLAGS.checkpoint_dir,
            optimizer=FLAGS.optimizer,
            mode=model_mode,
            strict=FLAGS.strict_context,
            pretrain_word_embed=FLAGS.pretrain_wordembed,
            typing=FLAGS.typing,
            el=FLAGS.el,
            coherence=FLAGS.coherence,
            textcontext=FLAGS.textcontext,
            useCNN=FLAGS.useCNN,
            WDLength=FLAGS.WDLength,
            Fsize=FLAGS.Fsize,
            entyping=FLAGS.entyping)

        if FLAGS.mode == 'inference':
            print("Doing inference")
            (predTypScNPmat_list,
             widIdxs_list,
             priorProbs_list,
             textProbs_list,
             jointProbs_list,
             evWTs_list,
             pred_TypeSetsList) = model.inference(ckptpath=FLAGS.model_path)

            numMentionsInference = len(widIdxs_list)
            numMentionsReader = 0
            for sent_idx in reader.sentidx2ners:
                numMentionsReader += len(reader.sentidx2ners[sent_idx])
            assert numMentionsInference == numMentionsReader

            mentionnum = 0
            entityTitleList = []
            for sent_idx in reader.sentidx2ners:
                nerDicts = reader.sentidx2ners[sent_idx]
                sentence = ' '.join(reader.sentences_tokenized[sent_idx])
                for s, ner in nerDicts:
                    [evWTs, evWIDS, evProbs] = evWTs_list[mentionnum]
                    predTypes = pred_TypeSetsList[mentionnum]
                    print(reader.bracketMentionInSentence(sentence, ner))
                    print("Prior: {} {}, Context: {} {}, Joint: {} {}".format(
                        evWTs[0], evProbs[0], evWTs[1], evProbs[1],
                        evWTs[2], evProbs[2]))

                    entityTitleList.append(evWTs[2])
                    print("Predicted Entity Types : {}".format(predTypes))
                    print("\n")
                    mentionnum += 1

            elview = copy.deepcopy(docta.view_dictionary['NER_CONLL'])
            elview.view_name = 'ENG_NEURAL_EL'
            for i, cons in enumerate(elview.cons_list):
                cons['label'] = entityTitleList[i]

            docta.view_dictionary['ENG_NEURAL_EL'] = elview

            print("elview.cons_list")
            print(elview.cons_list)
            print("\n")

            for v in docta.as_json['views']:
                print(v)
                print("\n")

        elif FLAGS.mode == 'test':
            print("Testing on Data ")
            (widIdxs_list, condProbs_list, contextProbs_list,
             condContextJointProbs_list, evWTs,
             sortedContextWTs) = model.dataset_test(ckptpath=FLAGS.model_path)

            print(len(widIdxs_list))
            print(len(condProbs_list))
            print(len(contextProbs_list))
            print(len(condContextJointProbs_list))
            print(len(reader.mentions))


            print("Writing Test Predictions: {}".format(FLAGS.test_out_fp))
            with open(FLAGS.test_out_fp, 'w') as f:
                for (wididxs, pps, mps, jps) in zip(widIdxs_list,
                                                    condProbs_list,
                                                    contextProbs_list,
                                                    condContextJointProbs_list):

                    mentionPred = ""

                    for (wididx, prp, mp, jp) in zip(wididxs, pps, mps, jps):
                        wit = reader.widIdx2WikiTitle(wididx)
                        mentionPred += wit + " " + str(prp) + " " + \
                            str(mp) + " " + str(jp)
                        mentionPred += "\t"

                    mentionPred = mentionPred.strip() + "\n"

                    f.write(mentionPred)

            print("Done writing. Can Exit.")

        else:
            print("WRONG MODE!")
            sys.exit(0)





    sys.exit()

if __name__ == '__main__':
    tf.app.run()
