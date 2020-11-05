import os
import sys
import copy
import json
import pprint
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join

from ccg_nlpy.core.view import View
from ccg_nlpy.core.text_annotation import TextAnnotation
from ccg_nlpy.local_pipeline import LocalPipeline
from readers.textanno_test_reader import TextAnnoTestReader
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

flags.DEFINE_string("input_jsonl", "", "Input containing documents in jsonl")
flags.DEFINE_string("output_jsonl", "", "Output in jsonl format")
flags.DEFINE_string("doc_key", "", "Key in input_jsonl containing documents")
flags.DEFINE_boolean("pretokenized", False, "Is the input text pretokenized")

FLAGS = flags.FLAGS

localpipeline = LocalPipeline()


def FLAGS_check(FLAGS):
    """
    Checks if the modelcontext

    Args:
        FLAGS: (int): write your description
    """
    if not (FLAGS.textcontext and FLAGS.coherence):
        print("*** Local and Document context required ***")
        sys.exit(0)
    assert os.path.exists(FLAGS.model_path), "Model path doesn't exist."

    assert(FLAGS.mode == 'ta'), "Only mode == ta allowed"


def main(_):
    """
    Main function.

    Args:
        _: (int): write your description
    """
    pp.pprint(flags.FLAGS.__flags)

    FLAGS_check(FLAGS)

    config = Config(FLAGS.config, verbose=False)
    vocabloader = VocabLoader(config)

    FLAGS.dropout_keep_prob = 1.0
    FLAGS.wordDropoutKeep = 1.0
    FLAGS.cohDropoutKeep = 1.0

    input_jsonl = FLAGS.input_jsonl
    output_jsonl = FLAGS.output_jsonl
    doc_key = FLAGS.doc_key

    reader = TextAnnoTestReader(
        config=config,
        vocabloader=vocabloader,
        num_cands=30,
        batch_size=FLAGS.batch_size,
        strict_context=FLAGS.strict_context,
        pretrain_wordembed=FLAGS.pretrain_wordembed,
        coherence=FLAGS.coherence)
    model_mode = 'test'

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

        model.load_ckpt_model(ckptpath=FLAGS.model_path)

        erroneous_files = 0

        outf = open(output_jsonl, 'w')
        inpf = open(input_jsonl, 'r')

        for line in inpf:
            jsonobj = json.loads(line)
            doctext = jsonobj[doc_key]
            ta = localpipeline.doc(doctext, pretokenized=FLAGS.pretokenized)
            _ = ta.get_ner_conll

            # Make instances for this document
            reader.new_ta(ta)

            (predTypScNPmat_list,
             widIdxs_list,
             priorProbs_list,
             textProbs_list,
             jointProbs_list,
             evWTs_list,
             pred_TypeSetsList) = model.inference_run()

            wiki_view = copy.deepcopy(reader.textanno.get_view("NER_CONLL"))
            docta = reader.textanno

            el_cons_list = wiki_view.cons_list
            numMentionsInference = len(widIdxs_list)

            assert len(el_cons_list) == numMentionsInference

            out_dict = {doc_key: doctext}
            el_mentions = []

            mentionnum = 0
            for ner_cons in el_cons_list:
                # ner_cons is a dict
                mentiondict = {}
                mentiondict['tokens'] = ner_cons['tokens']
                mentiondict['end'] = ner_cons['end']
                mentiondict['start'] = ner_cons['start']

                priorScoreMap = {}
                contextScoreMap = {}
                jointScoreMap = {}

                (wididxs, pps, mps, jps) = (widIdxs_list[mentionnum],
                                            priorProbs_list[mentionnum],
                                            textProbs_list[mentionnum],
                                            jointProbs_list[mentionnum])

                maxJointProb = 0.0
                maxJointEntity = ""
                for (wididx, prp, mp, jp) in zip(wididxs, pps, mps, jps):
                    wT = reader.widIdx2WikiTitle(wididx)
                    priorScoreMap[wT] = prp
                    contextScoreMap[wT] = mp
                    jointScoreMap[wT] = jp

                    if jp > maxJointProb:
                        maxJointProb = jp
                        maxJointEntity = wT

                mentiondict["jointScoreMap"] = jointScoreMap
                mentiondict["contextScoreMap"] = contextScoreMap
                mentiondict["priorScoreMap"] = priorScoreMap

                # add max scoring entity as label
                mentiondict["label"] = maxJointEntity
                mentiondict["score"] = maxJointProb

                mentionnum += 1

                el_mentions.append(mentiondict)

            out_dict['nel'] = el_mentions
            outstr = json.dumps(out_dict)
            outf.write(outstr)
            outf.write("\n")

        outf.close()
        inpf.close()

        print("Number of erroneous files: {}".format(erroneous_files))
        print("Annotation completed. Program can be exited safely.")
    sys.exit()

if __name__ == '__main__':
    tf.app.run()
