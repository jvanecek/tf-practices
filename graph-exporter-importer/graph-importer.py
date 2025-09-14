import tensorflow as tf
from tensorflow.python.platform import gfile

model_filename ='graph-int64.pb'

with tf.compat.v1.Session() as sess:
    print( '===== Before import =====' )
    print( sess.graph_def )
    
    with gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    print( '===== After import =====' )
    print( sess.graph_def ) 