import numpy as np
import tensorflow as tf
from math import exp, sqrt, pow, floor
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops import control_flow_ops
from tensorflow import random_normal_initializer as norm_init
from tensorflow import random_uniform_initializer as unif_init
from tensorflow import constant_initializer as const_init
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from cfg2 import lmdas, sigma_OUs
import os
import time

class worker(object):
    def __init__(self, sess):
        self.sess = sess
        self.T = 1
        self.gamma = 2
        self.r = .05

        self.year = 2014

        self.Winit = 1000
        self.transaction_fee = 0.032
        self.n_steps = 1600
        self.rescale_coeff = 1e6
        self.rescale_base = 0.999

        self.batch_size = 128
        self.num_path = 1000
        self._extra_train_ops = []
        self.Opt_value = 1000
        self.X_hist = []
        self.W_hist = []
        self.pi_hist = []
        self.pi_plus_hist = []
        self.dX_hist = []
        self.dW_hist = []
        self.lower_bound_hist = []
        self.upper_bound_hist = []

    def read_data(self): #read data from csv file
        df = pd.read_csv(os.getcwd()+'/merged.csv')
        df['Index'] = pd.to_datetime(df['Index'],format='%Y-%m-%d')
        df = df.set_index('Index')
        df = df.dropna()
        last_year_ind = np.logical_and(df.index >= datetime(self.year-1,1,1), df.index < datetime(self.year,1,1) )
        last_year_data = df[last_year_ind]
        this_year_ind = np.logical_and(df.index > datetime(self.year-1,12,31), df.index < datetime(self.year+1,1,1))
        this_year_data = df[this_year_ind]
        self.this_year_ind = [df.index[last_year_ind][-1]] + list(df.index[this_year_ind])

        self.Xinit = last_year_data.iloc[-1].get_values()
        self.d = self.Xinit.shape[0]
        self.last_year_days = len(last_year_data)

        self.N = len(this_year_data)



        self.sigma_OU = np.linalg.cholesky(sigma_OUs[str(self.year)]) * sqrt(self.last_year_days)
        self.lmda = lmdas[str(self.year)] * self.last_year_days
        self.sigma_c = self.sigma_OU.dot(self.sigma_OU)
        self.sigma_c_inv = np.linalg.inv(self.sigma_c)

        data_for_diff = this_year_data.append(last_year_data.iloc[-1])
        data_for_diff = data_for_diff.sort_index()
        diff_X = data_for_diff.diff().dropna()
        self.back_test_dX = diff_X.get_values().T
        self.lower_limit = -1
        self.upper_limit = 1
        self.mu = np.ones([self.d,1]) * 20
        self.n_neuron = [2, 20, 40, 80, self.d]



    def Utility(self, W): #rescaled utility function
        W1 = tf.maximum(W,.0001)
        return ((tf.pow(W1,1-self.gamma)-1) / (1-self.gamma)-self.rescale_base) * self.rescale_coeff

    def Solve_ABC(self): #Simulate A(tau), B(tau) and C(tau) function defined by the ODEs
        N = self.N
        d = self.d
        sigma_c = self.sigma_c
        sigma_c_inv = self.sigma_c_inv
        lmda = self.lmda
        mu = self.mu
        gamma = self.gamma
        r = self.r
        dt = self.T / self.N

        A = np.zeros([N + 1], dtype=float)
        B = np.zeros([d, N + 1], dtype=float)
        C = np.zeros([d, d, N + 1], dtype=float)
        for i in range(N):
            Ai = A[i]
            Bi = B[:, i:i + 1]
            Ci = C[:, :, i]
            dc = (1 - gamma) / gamma * Ci.T.dot(sigma_c).dot(Ci) + 1 / gamma * lmda.dot(sigma_c_inv).dot(lmda) + \
                 r / gamma * (lmda.dot(sigma_c_inv) + sigma_c_inv.dot(lmda)) + 1 / gamma * (r ** 2) * sigma_c_inv + \
                 r * (gamma - 1) / gamma * (Ci + Ci.T) - 1 / gamma * (Ci.T.dot(lmda) + lmda.dot(Ci))
            db = -r / gamma * sigma_c_inv.dot(lmda).dot(mu) - 1 / gamma * lmda.dot(sigma_c_inv).dot(lmda).dot(mu) \
                 + 1 / gamma * Ci.T.dot(lmda).dot(mu) + (1 - gamma) / gamma * Ci.T.dot(sigma_c).dot(Bi) \
                 + ((gamma - 1) / gamma * r * np.eye(d) - 1 / gamma * lmda).dot(Bi)
            da = 1 / gamma * Bi.T.dot(lmda).dot(mu) + 1 / (2 * gamma) * mu.T.dot(lmda).dot(sigma_c_inv).dot(lmda).dot(
                mu) \
                 + (1 - gamma) ** 2 / (2 * gamma) * Bi.T.dot(sigma_c).dot(Bi) + r + 1 / 2 * np.trace(
                sigma_c.dot(Ci + (1 - gamma) * Bi.dot(Bi.T)))

            A[i + 1] = Ai + da * dt
            B[:, i + 1:i + 2] = Bi + db * dt
            C[:, :, i + 1] = Ci + dc * dt

        self.A = A
        self.B = B
        self.C = C
        return A, B, C

    def Opt_Policy(self, tau, X): #Optimal policy under 0 transaction cost
        dt = self.T / self.N
        ind = int(tau / dt)
        t1 = tf.expand_dims(self.mu.reshape([1,-1]) - X,-1) #vector of (mu-X)
        t2 = tf.einsum('pj,ijk->ipk',tf.constant(self.lmda),t1) #vector of [lambda(\mu-X)]

        X_inv = tf.matrix_diag(1/X) #Diagonal matrix of 1/X
        mu_w = tf.matmul(X_inv,t2) #\mu_W defined in (19)
        Sigma_W = tf.einsum('ijk,kp->ijp',X_inv,tf.constant(self.sigma_OU))
        Sigma_W = tf.matmul(Sigma_W,Sigma_W,transpose_b=True) #\Sigma_W defined in lemma1
        sigma_ou_square = self.sigma_OU.dot(self.sigma_OU.T)
        Sigma_WX = tf.einsum('ijk,kp->ijp',X_inv,tf.constant(sigma_ou_square)) #\Sigma_{WX} defined in lemma 1
        B = self.B[:, ind:ind + 1]
        C = self.C[:, :, ind]
        C_dot_X = tf.einsum('pj,ijk->ipk',tf.constant(C),tf.expand_dims(X,axis=-1)) #vector of CX in (28)
        pi = -1 / self.gamma * tf.matmul(tf.matrix_inverse(Sigma_W) , (self.r-mu_w)-(1-self.gamma)*tf.matmul(Sigma_WX,tf.constant(B)+C_dot_X))
        return tf.squeeze(pi,axis=-1)

    def _one_time_net( self , x , bound, name ): #One neural network used to parametrize one bound at a particular time
        with tf.variable_scope( name ):
            x_norm = self._batch_norm(x, name = 'layer0_normal' )
            layer1 = self._one_layer( x_norm, self.n_neuron[1], activation_fn='relu',name = 'layer1' )
            layer2 = self._one_layer( layer1, self.n_neuron[2], activation_fn='relu',name = 'layer2' )
            layer3 = self._one_layer( layer2, self.n_neuron[3], activation_fn='relu',name = 'layer3' )
            z = self._one_layer( layer3, self.n_neuron[4], activation_fn='identity', name = 'final' )

        return z

    def _one_layer( self , input_, out_sz, activation_fn = 'ReLU', std =5.0 , name = 'linear' ): #One layer of neural network
        with tf.variable_scope( name ):
            shape = input_.get_shape().as_list()
            w = tf.get_variable( 'Matrix', [shape[1], out_sz], tf.float64,
                                 norm_init(stddev= std / np.sqrt(shape[1]+out_sz)) )
            hidden = tf.matmul( input_, w )
            hidden_bn = self._batch_norm( hidden, name = 'normal' )
        if activation_fn == 'relu': #Use leaky ReLU
            return tf.nn.relu(hidden_bn) + 0.2 * tf.nn.relu(-hidden_bn)
        else:
            return tf.nn.relu(hidden_bn) + 0.05 * tf.nn.relu(-hidden_bn)


    def _batch_norm( self, x, name ): #Batch normalization
        with tf.variable_scope( name ):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable( 'beta', params_shape, tf.float64, norm_init(0.0 , stddev =0.1, dtype = tf.float64) )
            gamma = tf.get_variable( 'gamma', params_shape, tf.float64, unif_init(.1, .5, dtype=tf.float64) )
            mv_mean = tf.get_variable( 'moving_mean', params_shape, tf.float64, const_init(0, tf.float64), trainable=False )
            mv_var = tf.get_variable( 'moving_variance', params_shape, tf.float64, const_init(1, tf.float64), trainable=False )

            mean, variance = tf.nn.moments(x, [0])
            self._extra_train_ops.append( assign_moving_average(mv_mean, mean, .99) )
            self._extra_train_ops.append( assign_moving_average(mv_var, variance, .99) )
            mean, variance = control_flow_ops.cond(self.is_training, lambda :(mean, variance),\
                                                   lambda : (mv_mean,mv_var))
            y = tf.nn.batch_normalization( x, mean, variance, beta, gamma, 1e-6 )
            y.set_shape(x.get_shape())
            return y


    def build(self): #Build the computational graph
        self.X = tf.placeholder(dtype = tf.float64,shape=[None,self.d])
        self.W = tf.placeholder(dtype = tf.float64, shape=[None,1])
        self.pi = tf.placeholder(dtype = tf.float64, shape=[None,self.d])
        self.dZ = tf.placeholder(dtype = tf.float64, shape=[None,self.d,self.N])
        self.W_pos = tf.placeholder(dtype = tf.bool, shape=[None,1])
        self.Play_Opt_Policy = tf.placeholder(dtype=tf.float64)
        self.is_training = tf.placeholder(tf.bool)
        self.is_back_test = tf.placeholder(dtype=tf.bool)
        X = self.X
        W = self.W
        pi = self.pi
        W_pos = self.W_pos
        self.loss = 0

        for i in range(self.N):
            tau = self.T - i / self.N * self.T
            lower_bound = self._one_time_net(X, bound=0.0,name = 'lower_bound_' + str(i))
            upper_bound = self._one_time_net(X, bound=0.0, name = 'upper_bound_' + str(i))
            lb = self.Opt_Policy(tau,X) - lower_bound * (1 - self.Play_Opt_Policy)
            ub = self.Opt_Policy(tau,X) + upper_bound * (1 - self.Play_Opt_Policy)
            print('Building step %d'%(i))

            pi_plus = tf.maximum(tf.minimum(pi,ub),lb) #Rebalance into the no trade zone
            pi_plus = tf.clip_by_value(pi_plus,self.lower_limit,self.upper_limit)  #Constrain the leverage ratio
            cost = tf.reduce_sum(tf.abs(pi-pi_plus),axis=-1,keep_dims=True) * W * self.transaction_fee #Transaction cost

            #Record values at different time step
            self.X_hist.append(X)
            self.W_hist.append(W)
            self.pi_hist.append(pi)
            self.pi_plus_hist.append(pi_plus)
            self.lower_bound_hist.append(lb)
            self.upper_bound_hist.append(ub)

            dX, dW = self.simulate_change(X,W,pi_plus,tf.expand_dims(self.dZ[:,:,i],-1)) #Calculate dX and dW
            self.dX_hist.append(dX)
            shares = pi_plus * W / X
            X = X + dX
            W = W + dW - cost
            pi = X * shares / W #Calculate \pi_{t-} for next time step
            W_pos = tf.logical_and(W_pos, W > 0) #Check if bankrupcy occurs on any path

        self.X_hist.append(X)
        self.W_hist.append(W)
        self.pi_hist.append(pi)
        self.pi_plus_hist.append(pi * 0.0)

        self.WT = W
        self.piT = pi

        W = W * tf.cast(W_pos, tf.float64) #if bankrupcy occurs on a paths, make its terminal wealth to be 0
        W = tf.maximum(W,0.0001) #make sure all the terminal wealth is positive, if bankrupcy occurs on a path its terminal wealth will be 0.0001 which gives a large penalization by utility function

        self.loss = tf.reduce_mean(self.Opt_value - self.Utility(W * (1- np.abs(pi) * self.transaction_fee))) #defining loss
        self.update_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker1')

    def simulate_change(self,Xj,Wj,pi_j,dZ):
        dt = self.T / self.N
        dX = []
        for i in range(self.d): #Simulate dX using equation (42)
            fct = exp(-self.lmda[i,i]*dt)
            dX_i = (self.mu[i]-Xj[:,i:i+1]) * (1-fct) + tf.squeeze(sqrt((1-fct**2) / (2 * self.lmda[i,i])) * tf.einsum('pj,ijk->ipk',tf.constant(self.sigma_OU[i:i+1,:]),dZ),axis=-1)
            dX.append(dX_i)

        dX = tf.concat(dX,axis=1)
        dX = tf.cond(self.is_back_test,lambda: tf.squeeze(dZ,axis=-1),lambda: dX) #if this is for back test, dX comes from the data input
        dW = tf.reduce_sum(pi_j * Wj / Xj * dX,axis=1,keep_dims=True) + (1-tf.reduce_sum(pi_j,axis=1,keep_dims=True)) * Wj * (exp(self.r*dt)-1) #calculate dW using equation (44)
        return dX, dW




    def process_save_data(self, x, w, p, p_plus, dx, lb, ub):
        x = np.array(x).reshape([self.N+1,self.num_path,self.d])
        w = np.array(w).reshape([self.N+1,self.num_path])
        p = np.array(p).reshape([self.N+1,self.num_path,self.d])
        p_plus = np.array(p_plus).reshape([self.N+1,self.num_path,self.d])
        dx = np.array(dx).reshape([self.N,self.num_path,self.d])
        lb = np.array(lb).reshape([self.N,self.num_path,self.d])
        ub = np.array(ub).reshape([self.N,self.num_path,self.d])
        np.save(str(self.year)+'.npy',{
                                'x' : x,
                                'w' : w,
                                'p' : p,
                                'p_plus' : p_plus,
                                'dx' : dx,
                                'lb' : lb,
                                'ub' : ub})



    def train(self):
        sess = self.sess
        trainable_variables = tf.trainable_variables()
        self.global_step = tf.get_variable('global_step', [], initializer=const_init(1), trainable=False,
                                           dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(1.0, self.global_step, decay_steps=300, decay_rate=0.5,staircase=False)
        grads = tf.gradients(self.loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer( learning_rate=learning_rate  )
        apply_op = optimizer.apply_gradients( zip(grads, trainable_variables), global_step=self.global_step, name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        train_op = tf.group(*train_ops)
        multiplier = 30


        sess.run(tf.global_variables_initializer())

        self.num_path = 100 * multiplier


        step, loss, X_hist, W_hist, pi_hist, pi_plus_hist, dX_hist, dW_hist, \
        lower_bound_hist, upper_bound_hist = sess.run([self.global_step, self.loss, \
                                                       self.X_hist, self.W_hist, self.pi_hist, self.pi_plus_hist, \
                                                       self.dX_hist, self.dW_hist, self.lower_bound_hist,
                                                       self.upper_bound_hist], feed_dict={
            self.X: np.ones([self.num_path, 1]).dot(self.Xinit.reshape([1, -1])),
            self.W: np.ones([self.num_path, 1]) * self.Winit,
            self.pi: np.zeros([self.num_path, self.d]),
            self.dZ: np.random.normal(size=[self.num_path, self.d, self.N]),
            self.W_pos: np.ones([self.num_path, 1], dtype=bool),
            self.Play_Opt_Policy: 1.0,
            self.is_back_test: False,
            self.is_training: False})
        print('Playing Optimal Policy: %f' %(np.mean(sess.run(self.Utility(W_hist[-1])))))


        for i in range(self.n_steps):
            sess.run(train_op,feed_dict={
                    self.X : np.ones([self.num_path,1]).dot(self.Xinit.reshape([1,-1])),
                    self.W : np.ones([self.num_path,1]) * self.Winit,
                    self.pi : np.zeros([self.num_path,self.d]),
                    self.dZ : np.random.normal(size=[self.num_path,self.d,self.N]),
                    self.W_pos: np.ones([self.num_path,1],dtype=bool),
                    self.Play_Opt_Policy: 0.0,
                    self.is_back_test: False,
                    self.is_training : True})

            if i % 50 == 0 or i == self.n_steps - 1:

                step, loss, X_hist, W_hist, pi_hist, pi_plus_hist, dX_hist, dW_hist, \
                    lower_bound_hist, upper_bound_hist = sess.run([self.global_step,self.loss, \
                                                                   self.X_hist, self.W_hist, self.pi_hist, self.pi_plus_hist, \
                                                                   self.dX_hist, self.dW_hist, self.lower_bound_hist, self.upper_bound_hist],feed_dict={
                    self.X: np.ones([self.num_path, 1]).dot(self.Xinit.reshape([1,-1])),
                    self.W: np.ones([self.num_path, 1]) * self.Winit,
                    self.pi: np.zeros([self.num_path, self.d]),
                    self.dZ: np.random.normal(size=[self.num_path, self.d, self.N]),
                    self.W_pos: np.ones([self.num_path, 1], dtype=bool),
                    self.Play_Opt_Policy: 0.0,
                    self.is_back_test: False,
                        self.is_training : False})
                print('step = %d, loss = %f' %(step, loss))
                if loss < 800:
                    print('good result!')


        self.num_path = 1
        step, loss, X_hist, W_hist, pi_hist, pi_plus_hist, dX_hist, dW_hist, \
        lower_bound_hist, upper_bound_hist = sess.run([self.global_step, self.loss, \
                                                       self.X_hist, self.W_hist, self.pi_hist, self.pi_plus_hist, \
                                                       self.dX_hist, self.dW_hist, self.lower_bound_hist,
                                                       self.upper_bound_hist], feed_dict={
            self.X: np.ones([self.num_path, 1]).dot(self.Xinit.reshape([1, -1])),
            self.W: np.ones([self.num_path, 1]) * self.Winit,
            self.pi: np.zeros([self.num_path, self.d]),
            self.dZ: np.expand_dims(self.back_test_dX,axis=0),
            self.W_pos: np.ones([self.num_path, 1], dtype=bool),
            self.Play_Opt_Policy: 0.0,
            self.is_back_test: True,
            self.is_training: False})

        self.process_save_data(X_hist, W_hist, pi_hist, pi_plus_hist, dX_hist, lower_bound_hist,
                               upper_bound_hist)
        print("finished")

with tf.Session() as sess:
    tf.set_random_seed(2)
    np.random.seed(2)
    model = worker(sess)
    model.read_data()
    model.Solve_ABC()
    # model.Play_Opt_Policy()
    model.build()
    model.train()
