import unittest
import numpy as np
import tensorflow as tf

class TestTensorflow(unittest.TestCase):

    def testFeedingPlaceholdersInDifferentOperations(self): 
        W = tf.constant([10,100], name='const_W')
        #these placeholders can hold tensors of any shape
        #we will feed these placeholders later
        x = tf.placeholder(tf.int32, name='x')
        b = tf.placeholder(tf.int32,name='b')
        #tf.multiply is simple multiplication and not matrix
        Wx = tf.multiply(W,x, name="Wx")
        y = tf.add(Wx,b,name='y')
 
        with tf.Session() as sess:
            '''all the code which require a session is writer here
            here Wx is the fetches parameter. fetches refers to the node of the graph we want to compute
            feed_dict is used to pass the values for the placeholders
            '''
            self.assertListEqual( sess.run(Wx, feed_dict={x: [3,33]}).tolist(), [10*3, 100*33] )
            self.assertListEqual( sess.run(y, feed_dict={x:[5,50],b:[7,9]}).tolist(), [10*5+7, 100*50+9] )
            
    def testFeedingPlaceholders(self):
        W = tf.Variable([2.5,4.0],tf.float32, name='var_W')
        x = tf.placeholder(tf.float32, name='x')
        b = tf.Variable([5.0,10.0],tf.float32, name='var_b')
        y = W * x + b

        #initialize all variables defined
        init = tf.compat.v1.global_variables_initializer()
        #global_variable_initializer() will declare all the variable we have initilized
        # use with statement to instantiate and assign a session
        with tf.Session() as sess:
            sess.run(init)
            #this computation is required to initialize the variable
            self.assertListEqual( sess.run(y,feed_dict={x:[10,100]}).tolist(),  [2.5*10+5,4*100+10] )

    def testIncrementingOfVariableInOperation(self):
        number = tf.Variable(2)
        multiplier = tf.Variable(1)
        init = tf.compat.v1.global_variables_initializer()
        result = number.assign(tf.multiply(number,multiplier))
        with tf.Session() as sess:
            sess.run(init)
            for i in range(10):
                self.assertEqual(sess.run(result), np.math.factorial(i+1)*2 )
                self.assertEqual(sess.run(multiplier.assign_add(1)), i+2 )

    def testCreatingMultipleGraphs(self): 
        g1 = tf.Graph()
        '''set g1 as default to add tensors to this graph using default methord'''
        with g1.as_default():
            with tf.Session() as sess:
                A = tf.constant([5,7],tf.int32, name='A')
                x = tf.placeholder(tf.int32, name='x')
                b = tf.constant([3,4],tf.int32, name='b')
                y = A * x + b
                self.assertListEqual( sess.run(y, feed_dict={x: [10,100]}).tolist(), [53, 704] )
                '''to ensure all the tensors and computations are within the graph g1, we use assert'''
            self.assertEqual( y.graph, g1 ) 

        g2 = tf.Graph()
        with g2.as_default():
            with tf.Session() as sess:
                A = tf.constant([5,7],tf.int32, name='A')
                x = tf.placeholder(tf.int32, name='x')
                y = tf.pow(A,x,name='y')
                self.assertListEqual( sess.run(y, feed_dict={x: [3,5]}).tolist(), [125, 16807] )
            self.assertEqual( y.graph, g2 ) 

        '''same way you can access defaut graph '''
        default_graph = tf.compat.v1.get_default_graph()
        with tf.Session() as sess:
            A = tf.constant([5,7],tf.int32, name='A')
            x = tf.placeholder(tf.int32, name='x')
            y = A + x
            self.assertListEqual(sess.run(y, feed_dict={x: [3,5]}).tolist(), [8,12])
        self.assertEqual( y.graph,  default_graph ) 

    " Following examples based on https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/GradientTape "
    def testGradients(self):
        x = tf.ones((2, 2))

        with tf.Session() as sess:
            with tf.GradientTape() as t:
                t.watch(x)
                y = tf.reduce_sum(x)
                z = tf.multiply(y, y)

                # Derivative of z with respect to the original input tensor x
                dz_dx = t.gradient(z, x)
                for i in [0, 1]:
                    for j in [0, 1]:
                        self.assertEqual( sess.run( dz_dx[i][j] ), 8.0 )

    def _withPersistentTapeDo(self, anAssertion): 
        with tf.GradientTape(persistent=True) as t:
            anAssertion(t)
        del t # Drop the reference to the tape

    def testComputeMultipleGradientsOverSameComputation(self): 
        x = tf.constant(3.0)

        def assertionTested(t):
            t.watch(x)
            y = x * x
            z = y * y
            dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
            dy_dx = t.gradient(y, x)  # 6.0
            
            with tf.Session() as sess:
                self.assertEqual( sess.run( dz_dx ), 108 )
                self.assertEqual( sess.run( dy_dx ), 6 )
        
        " In order to compute multiple gradients in same computation, use a persistent gradient tape"
        self._withPersistentTapeDo(assertionTested)

    
    def testComputeMultipleGradientsOverSameComputationFeedingOperations(self): 
        x = tf.placeholder(tf.float32, name='x')
        
        def assertionTested(t):
            t.watch(x)
            y = x * x
            z = y * y
            dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
            dy_dx = t.gradient(y, x)  # 6.0
            
            with tf.Session() as sess:
                self.assertEqual( sess.run( dz_dx, feed_dict={x: [3,]} ), 108 )
                self.assertEqual( sess.run( dy_dx, feed_dict={x: [3,]} ), 6 )
        
        self._withPersistentTapeDo(assertionTested)

    def testGradientsAreUnavailableUsingPlaceholdersOfInt32(self): 
        x = tf.placeholder(tf.int32, name='x')
        
        with tf.GradientTape() as t:
            t.watch(x)
            y = x * x
            z = y * y
            dz_dx = t.gradient(z, x) 
            
            # Using gradients with int32 fails quietly
            self.assertEqual( dz_dx, None )

    def testHighOrderDerivativesUsingNestedGradients(self): 
        x = tf.constant(3.0)
        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                y = x * x
            dy_dx = gg.gradient(y, x)     # Will compute to 6.0
        d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0

        with tf.Session() as sess:
            self.assertEqual( sess.run( dy_dx ), 6.0)
            self.assertEqual( sess.run( d2y_dx2 ), 2.0)
    
    def testHighOrderDerivativesUsingSingleGradientTape(self): 
        x = tf.constant(3.0)
        
        def assertionTested(t):
            t.watch(x)
            y = x * x
            dy_dx = t.gradient(y, x)
            d2y_dx2 = t.gradient(dy_dx, x)

            with tf.Session() as sess:
                self.assertEqual( sess.run( dy_dx ), 6.0)
                self.assertEqual( sess.run( d2y_dx2 ), 2.0)
        
        self._withPersistentTapeDo(assertionTested)

if __name__ == '__main__':
    unittest.main()
