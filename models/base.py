import os
from glob import glob
import tensorflow as tf

class Model(object):
    """Abstract object representing an Reader model."""
    def __init__(self):
        pass

    # def get_model_dir(self):
    #   model_dir = self.dataset
    #   for attr in self._attrs:
    #     if hasattr(self, attr):
    #       model_dir += "/%s=%s" % (attr, getattr(self, attr))
    #   return model_dir

    def get_model_dir(self, attrs=None):
        model_dir = self.dataset
        if attrs == None:
            attrs = self._attrs
        for attr in attrs:
            if hasattr(self, attr):
                model_dir += "/%s=%s" % (attr, getattr(self, attr))
        return model_dir

    def get_log_dir(self, root_log_dir, attrs=None):
        model_dir = self.get_model_dir(attrs=attrs)
        log_dir = os.path.join(root_log_dir, model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save(self, saver, checkpoint_dir, attrs=None, global_step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__
        model_dir = self.get_model_dir(attrs=attrs)

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                   global_step=global_step)
        print(" [*] Saving done...")

    def initialize(self, log_dir="./logs"):
        self.merged_sum = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph_def)

        tf.initialize_all_variables().run()
        self.load(self.checkpoint_dir)

        start_iter = self.step.eval()

    def load(self, saver, checkpoint_dir, attrs=None):
        print(" [*] Loading checkpoints...")
        model_dir = self.get_model_dir(attrs=attrs)
        # /checkpointdir/attrs=values/
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print(" [#] Checkpoint Dir : {}".format(checkpoint_dir))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("ckpt_name: {}".format(ckpt_name))
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [!] Load failed...")
            return False

    def loadCKPTPath(self, saver, ckptPath=None):
        assert ckptPath != None
        print(" [#] CKPT Path : {}".format(ckptPath))
        if os.path.exists(ckptPath):
            saver.restore(self.sess, ckptPath)
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [*] CKPT Path doesn't exist")
            return False

    def loadSpecificCKPT(self, saver, checkpoint_dir, ckptName=None, attrs=None):
        assert ckptName != None
        model_dir = self.get_model_dir(attrs=attrs)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint_path = os.path.join(checkpoint_dir, ckptName)
        print(" [#] CKPT Path : {}".format(checkpoint_path))
        if os.path.exists(checkpoint_path):
            saver.restore(self.sess, checkpoint_path)

            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [*] CKPT Path doesn't exist")
            return False



    def collect_scope(self, scope_name, graph=None, var_type=tf.GraphKeys.VARIABLES):
        if graph == None:
            graph = tf.get_default_graph()

        var_list = graph.get_collection(var_type, scope=scope_name)

        assert_str = "No variable exists with name_scope '{}'".format(scope_name)
        assert len(var_list) != 0, assert_str

        return var_list

    def get_scope_var_name_set(self, var_name):
        clean_var_num = var_name.split(":")[0]
        scopes_names = clean_var_num.split("/")
        return set(scopes_names)


    def scope_vars_list(self, scope_name, var_list):
        scope_var_list = []
        for var in var_list:
            scope_var_name = self.get_scope_var_name_set(var.name)
            if scope_name in scope_var_name:
                scope_var_list.append(var)
        return scope_var_list
