import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        # TODO Failed on GPU...
        worker_device="/job:worker/task:%d/cpu:0" % FLAGS.task_index,
        ps_device="/job:ps/cpu:0",
        #worker_device="/job:worker/task:%d/cpu:0" % FLAGS.task_index,
        cluster=cluster)):
    
      global_step = tf.contrib.framework.get_or_create_global_step()

      # Build model...
      with tf.name_scope("input"):
        mnist = input_data.read_data_sets("./input_data", one_hot=True)
        x = tf.placeholder(tf.float32, [None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")
      
      tf.set_random_seed(1)
      with tf.name_scope("weights"):
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

      with tf.name_scope("model"):
        y = tf.matmul(x, W) + b

      with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

      with tf.name_scope("train"):
        #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        opt = tf.train.GradientDescentOptimizer(0.5)
        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=1,
            total_num_replicas=1,
            name="mnist_sync_replicas")
        train_step = opt.minimize(cross_entropy, global_step=global_step)

      with tf.name_scope("acc"):
        init_op = tf.initialize_all_variables()
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      local_init_op = opt.chief_init_op
      ready_for_local_init_op = opt.ready_for_local_init_op
      chief_queue_runner = opt.get_chief_queue_runner()
      sync_init_op = opt.get_init_tokens_op()
      init_op = tf.global_variables_initializer()

      sv = tf.train.Supervisor(
        init_op=init_op,
        local_init_op=local_init_op,
        ready_for_local_init_op=ready_for_local_init_op,
        recovery_wait_secs=1,
        global_step=global_step)
      
      config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False,
          device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
      
      sess = sv.prepare_or_wait_for_session(server.target, config=config)
      print("Worker %d: Session initialization complete." % FLAGS.task_index)
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    # The StopAtStepHook handles stopping after running given steps.
    #hooks=[tf.train.StopAtStepHook(last_step=1000000)]


    #with tf.train.MonitoredTrainingSession(master=server.target,
    #                                       #config=config,
    #                                       is_chief=(FLAGS.task_index == 0),
    #                                       #checkpoint_dir="/tmp/train_logs",
    #                                       hooks=hooks) as mon_sess:
      for _ in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      print("accuracy is: ")
      print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



#      loss = ...
#      global_step = tf.contrib.framework.get_or_create_global_step()
#
#      train_op = tf.train.AdagradOptimizer(0.01).minimize(
#          loss, global_step=global_step)
#
#    # The StopAtStepHook handles stopping after running given steps.
#    hooks=[tf.train.StopAtStepHook(last_step=1000000)]
#
#    # The MonitoredTrainingSession takes care of session initialization,
#    # restoring from a checkpoint, saving to a checkpoint, and closing when done
#    # or an error occurs.
#    with tf.train.MonitoredTrainingSession(master=server.target,
#                                           is_chief=(FLAGS.task_index == 0),
#                                           checkpoint_dir="/tmp/train_logs",
#                                           hooks=hooks) as mon_sess:
#      while not mon_sess.should_stop():
#        # Run a training step asynchronously.
#        # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
#        # perform *synchronous* training.
#        # mon_sess.run handles AbortedError in case of preempted PS.
#        mon_sess.run(train_op)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)