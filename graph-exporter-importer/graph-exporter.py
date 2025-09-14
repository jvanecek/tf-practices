import tensorflow as tf

# Define the graph
graph = tf.Graph()
with graph.as_default():
    a = tf.constant(42, dtype=tf.int64, name='a')

# Save the graph to a protobuf file
with tf.compat.v1.Session(graph=graph) as sess:
    tf.io.write_graph(sess.graph_def, '.', 'graph-int64.pb', as_text=False)
