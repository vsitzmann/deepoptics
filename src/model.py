import os
from pprint import pprint

import tensorflow as tf
import abc

slim = tf.contrib.slim

class Model(abc.ABC):
    """Generic tensorflow model class.
    """
    def __init__(self, name, ckpt_path=None):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=sess_config)

        self.name = name
        self.ckpt_path = ckpt_path

    @abc.abstractmethod
    def _build_graph(self, x_train, global_step, **kwargs):
        """Builds the model, given x_train as input.

        Args:
            x_train: The dequeued training example
            **kwargs: Model parameters that can later be passed to the "fit" function

        Returns:
            model_output: The output of the model
        """

    @abc.abstractmethod
    def _get_data_loss(self,
                      model_output,
                      ground_truth):
        """Computes the data loss (not regularization loss) of the model.

        For consistency of weighing of regularization loss vs. data loss,
        normalize loss by batch size.

        Args:
            model_output: Output of self._build_graph
            ground_truth: respective ground truth

        Returns:
            data_loss: Scalar data loss of the model.         """


    def _get_reg_loss(self):
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return reg_loss


    @abc.abstractmethod
    def _get_training_queue(self, batch_size):
        """Builds the queues for training data.

        Use tensorflow's readers, decoders and tf.train.batch to build the dataset.

        Args:
            batch_size:

        Returns:
            x_train: the dequeued model input
            y_train: the dequeued ground truth
        """

    def _set_up_optimizer(self,
                          starter_learning_rate,
                          decay_type,
                          decay_params,
                          opt_type,
                          opt_params,
                          global_step):
        if decay_type is not None:
            if decay_type == 'exponential':
                learning_rate = tf.train.exponential_decay(starter_learning_rate[key],
                                                           global_step,
                                                           **decay_params)
            elif decay_type == 'polynomial':
                learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                                           global_step,
                                                           **decay_params)
        else:
            learning_rate = starter_learning_rate

        if opt_type == 'ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               **opt_params)
        elif opt_type == 'sgd_with_momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   **opt_params)
        elif opt_type == 'Adadelta' or opt_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                                   **opt_params)
        elif opt_type == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  **opt_params)
        else:
            raise Exception('Unknown opt type')

        tf.summary.scalar('learning_rate', learning_rate)
        return optimizer



    def fit(self,
            model_params, # Dictionary of model parameters
            opt_type, # Type of optimization algorithm
            opt_params, # Parameters of optimization algorithm
            batch_size,
            starter_learning_rate,
            logdir,
            num_steps,
            num_steps_until_save,
            num_steps_until_summary,
            num_steps_until_val=None,
            x_val_list=None,
            decay_type=None, # Type of decay
            decay_params=None, # Decay parameters
            feed_dict=None,
            ):
        """Trains the model.
        """
        x_train, y_train = self._get_training_queue(batch_size)

        print("\n\n")
        print(40*"*")
        print("Saving model and summaries to %s"%logdir)
        print("Optimization parameters:")
        print(opt_type)
        print(opt_params)
        print("Starter learning rate/s is/are ",starter_learning_rate)
        print("Model parameters:")
        print(model_params)
        print(40*"*")
        print("\n\n")

        global_step = tf.Variable(0, trainable=False)

        model_output_train = self._build_graph(x_train, global_step, **model_params)
        data_loss_graph = self._get_data_loss(model_output_train, y_train)
        reg_loss_graph = self._get_reg_loss()
        total_loss_graph = tf.add(reg_loss_graph,
                                  data_loss_graph)

        all_variables = tf.trainable_variables()

        if isinstance(starter_learning_rate, dict):
            training_steps = []

            for key in starter_learning_rate:
                # Get the list of variables
                optimizer = self._set_up_optimizer(starter_learning_rate[key],
                                                    decay_type[key],
                                                    decay_params[key],
                                                    opt_type[key],
                                                    opt_params[key],
                                                    global_step=global_step)

                var_list = [var for var in all_variables if key in var.name]
                print("Single optimizer for following group of variables:")
                pprint(var_list)

                train_step = slim.learning.create_train_op(total_loss_graph,
                                                           optimizer=optimizer,
                                                           global_step=global_step,
                                                           variables_to_train=var_list,
                                                           summarize_gradients=True)
                training_steps.append(train_step)

            train_step = tf.group(*training_steps)
        else:
            optimizer = self._set_up_optimizer(starter_learning_rate,
                                               decay_type,
                                               decay_params,
                                               opt_type,
                                               opt_params,
                                               global_step=global_step)
            train_step = slim.learning.create_train_op(total_loss_graph,
                                                       optimizer=optimizer,
                                                       variables_to_train=all_variables,
                                                       global_step=global_step,
                                                       summarize_gradients=True)

        # Attach summaries to salient training parameters
        tf.summary.scalar('data_loss', data_loss_graph)
        tf.summary.scalar('reg_loss', reg_loss_graph)
        tf.summary.scalar('total_loss', total_loss_graph)

        # Create a saver for saving out the models
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2,
                max_to_keep=5)

        # Create saver to restore variables from checkpoints
        all_variables = set(slim.get_model_variables() + tf.trainable_variables())

        if self.ckpt_path is not None:
            if isinstance(self.ckpt_path, dict):
                restoration_savers = {}
                for var_id in self.ckpt_path:
                    variables = [var for var in all_variables if var_id in var.name]
                    print("Restoring the following variables:")
                    pprint([var.name for var in variables])
                    restoration_savers[var_id] = tf.train.Saver(variables)
            else:
                restoration_saver = tf.train.Saver()

        # Get all summaries
        summaries_merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir, self.sess.graph, flush_secs=60)

        # Init op
        init = tf.global_variables_initializer()
        self.sess.run(init,feed_dict=feed_dict)

        if self.ckpt_path is not None:
            if isinstance(self.ckpt_path, dict):
                for var_id in self.ckpt_path:
                    print("Loading from checkpoint path %s"%self.ckpt_path)
                    restoration_savers[var_id].restore(self.sess, self.ckpt_path[var_id])
            else:
                restoration_saver.restore(self.sess, self.ckpt_path)

        # Train the model
        print("Beginning the training")
        for step in range(num_steps):
            try:
                _, total_loss, reg_loss, data_loss = self.sess.run([train_step,
                                                                    total_loss_graph,
                                                                    reg_loss_graph,
                                                                    data_loss_graph],
                                                                    feed_dict=feed_dict)
                print("Step %d\n    total_loss %0.8f   reg_loss %0.8f   data_loss %0.8f\n"%\
                        (step, total_loss, reg_loss, data_loss))

                if not step % num_steps_until_save and step:
                    print("Saving model...")
                    save_path = os.path.join(logdir, self.name+'.ckpt')

                    if decay_type is not None:
                        self.saver.save(self.sess, save_path, global_step=global_step)
                    else:
                        self.saver.save(self.sess, save_path, global_step=step)

                if not step % num_steps_until_summary:
                    print("Writing summaries...")
                    summary = self.sess.run(summaries_merged, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, step)

            except tf.errors.OutOfRangeError:
                print("Dataset reported end of repetitions.")

        print("Saving final model.")
        save_path = os.path.join(logdir, self.name+'.ckpt')

        if decay_type is not None:
            self.saver.save(self.sess, save_path, global_step=global_step)
        else:
            self.saver.save(self.sess, save_path, global_step=step)
