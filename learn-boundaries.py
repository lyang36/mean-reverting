import numpy as np
import tensorflow as tf
from math import exp, sqrt, pow, floor
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops import control_flow_ops
from tensorflow import random_normal_initializer as norm_init
from tensorflow import random_uniform_initializer as unif_init
from tensorflow import constant_initializer as const_init
from tensorflow.contrib.layers import dropout
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import time

class worker(object):
    def __init__(self, sess):
        self.sess = sess
        self.T = 1
        self.mu = 15.4463
        self.sigma = 0.6055269 * sqrt(252)
        self.gamma = 2
        self.lmda = 0.1127820 * 252
        self.r = .05
        self.N = 50
        self.limit = 1

        self.n_neuron = [1, 20, 40, 80, 1]

        self.Xinit = 13.949999999999999
        self.Winit = 1000
        self.transaction_fee = 0.020
        self.n_steps = 3000
        self.rescale_coeff = 1e6
        self.rescale_base = 0.999

        #coefficients for calculating Btau and Ctau
        self.a0 = self.sigma ** 2 * (1 - self.gamma) / self.gamma
        self.b0 = 2 * (self.gamma * self.r - self.r - self.lmda) / self.gamma
        self.c0 = (self.lmda + self.r) ** 2 / (self.sigma ** 2 * self.gamma)
        self.d0 = self.lmda * self.mu / self.gamma
        self.e0 = - self.lmda * self.mu * (self.lmda + self.r) / (self.sigma ** 2 * self.gamma)
        self.eta = sqrt(self.b0 ** 2 - 4 * self.a0 * self.c0)

        self.num_path = 10000
        self._extra_train_ops = []
        self.Opt_value = 1000
        self.X_hist = [] #history of risky asset price
        self.W_hist = [] #history of wealth
        self.pi_hist = [] #history of position before rebalancing
        self.pi_plus_hist = [] #history of position after rebalancing
        self.dX_hist = [] #history of change of price dX
        self.lower_bound_hist = [] #history of lower boundary
        self.upper_bound_hist = [] #history of upper boundary

    def Ctau(self,tau): # C(tau) function
        numerator = 2 * self.c0 * (1-np.exp(-self.eta * tau))
        denom = 2 * self.eta - (self.b0+self.eta) * (1-np.exp(-self.eta * tau))
        return numerator / denom

    def Btau(self,tau): # B(tau) function
        numerator = -4 * self.e0 * self.r * (np.square(1-np.exp(-self.eta * tau / 2))) \
                    + 2 * self.e0 * self.eta * (1-np.exp(-self.eta*tau))
        denom = self.eta * (2 * self.eta - (self.b0+self.eta)*(1-np.exp(-self.eta*tau)))
        return numerator / denom

    def Opt_Policy(self,tau, X): #Optimal policy under 0 transaction cost
        C = self.Ctau(tau)
        B = self.Btau(tau)
        return (-(self.lmda+self.r)/(self.sigma**2*self.gamma)+(1-self.gamma)/self.gamma * C) * np.square(X) \
                + ((1-self.gamma)/self.gamma*B+self.lmda*self.mu/(self.sigma**2*self.gamma)) * X

    def Play_Opt_Policy(self): #Simulation of trading using policy derived under 0 transaction cost
        X = np.ones([self.num_path,1]) * self.Xinit
        W = np.ones([self.num_path,1]) * self.Winit
        dZ = np.random.normal(size=[self.num_path,self.N])
        pi = np.zeros([self.num_path,1])
        ui = [] #history of mean utility over paths
        wi = [] #history of mean wealth over paths

        for i in range(self.N):
            tau = (self.N-i) / self.N * self.T
            pos = np.minimum(np.maximum(self.Opt_Policy(tau,X),-1),1)
            dX, dW = self.simulate_change(X,W,pos,dZ[:,i].reshape([-1,1]))
            cost = np.abs(pos-pi) * W * self.transaction_fee
            ui.append(np.mean(sess.run(self.Utility(W))))
            wi.append(np.mean(W))
            X = X + dX
            W = W + dW - cost
            pi = X / W

        ui.append(np.mean(sess.run(self.Utility(W))))
        wi.append(np.mean(W))
        print('Mean utility over time: ', ui)
        print('Mean wealth over time: ', wi)
        self.opt_policy_utility = ui
        self.opt_policy_wealth = wi



    def Utility(self, W): # Rescaled utility function
        W1 = tf.maximum(W,.0001)
        return ((np.power(W1,1-self.gamma)-1) / (1-self.gamma)-self.rescale_base) * self.rescale_coeff


    def _one_time_net( self , x , bound, name ): # Building neural net at each time step
        with tf.variable_scope( name ):
            x_norm = self._batch_norm(x, name = 'layer0_normal' )
            layer1 = self._one_layer( x_norm, self.n_neuron[1], activation_fn='relu',name = 'layer1' )
            layer2 = self._one_layer( layer1, self.n_neuron[2], activation_fn='relu',name = 'layer2' )
            layer3 = self._one_layer( layer2, self.n_neuron[3], activation_fn='relu',name = 'layer3' )
            z = self._one_layer( layer3, self.n_neuron[4], activation_fn='identity', name = 'final' )
            z = tf.maximum(z,bound)
        return z

    def _one_layer( self , input_, out_sz, activation_fn = 'ReLU', std =5.0 , name = 'linear' ): # Building one layer of neural net
        with tf.variable_scope( name ):
            shape = input_.get_shape().as_list()
            w = tf.get_variable( 'Matrix', [shape[1], out_sz], tf.float64,
                                 norm_init(stddev= std / np.sqrt(shape[1]+out_sz)) )
            hidden = tf.matmul( input_, w )
            hidden_bn = self._batch_norm( hidden, name = 'normal' )

        if activation_fn == 'relu': # Leaky ReLU function to avoid dying ReLU
            return tf.nn.relu(hidden_bn) + 0.2 * tf.nn.relu(-hidden_bn)
        else:
            return hidden_bn


    def _batch_norm( self, x, name ): # Batch normalization
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


    def build(self):
        self.X = tf.placeholder(dtype = tf.float64,shape=[None,1])
        self.W = tf.placeholder(dtype = tf.float64, shape=[None,1])
        self.pi = tf.placeholder(dtype = tf.float64, shape=[None,1])
        self.dZ = tf.placeholder(dtype = tf.float64, shape=[None,self.N])
        self.W_pos = tf.placeholder(dtype = tf.bool, shape=[None,1]) # Indicator function of the paths has not seen a bankruptcy, W_pos means W is positive
        self.lower_limit = tf.placeholder(dtype = tf.float64, shape = [None,1]) # leverage ratio of short
        self.upper_limit = tf.placeholder(dtype = tf.float64, shape = [None,1]) # leverage ratio of long
        self.is_training = tf.placeholder(tf.bool)

        X = self.X
        W = self.W
        pi = self.pi
        W_pos = self.W_pos

        for i in range(self.N):
            tau = i / self.N * self.T
            lower_bound = self._one_time_net(X, bound=0.0,name = 'lower_bound_' + str(i))
            upper_bound = self._one_time_net(X, bound=0.0, name = 'upper_bound_' + str(i))
            lb = self.Opt_Policy(tau,X) - lower_bound
            ub = self.Opt_Policy(tau,X) + upper_bound
            pi_plus = tf.maximum(tf.minimum(pi,ub),lb) # rebalance the position into the non-trading zone
            pi_plus = tf.clip_by_value(pi_plus,self.lower_limit,self.upper_limit) # ensure the leverage ratio is within [-1,1]
            cost = tf.abs(pi-pi_plus) * W * self.transaction_fee # linear cost due to rebalancing
            self.X_hist.append(X)
            self.W_hist.append(W)
            self.pi_hist.append(pi)
            self.pi_plus_hist.append(pi_plus)
            self.lower_bound_hist.append(lb)
            self.upper_bound_hist.append(ub)
            dX, dW = self.simulate_change(X,W,pi_plus,tf.expand_dims(self.dZ[:,i],-1))
            self.dX_hist.append(dX)
            shares = pi_plus * W / X # current holding shares of risky asset
            X = X + dX
            W = W + dW - cost
            pi = X * shares / W # proportion invested in risky asset at the beginning of next stage
            W_pos = tf.logical_and(W_pos, W > 0) # check whether bankrupcy occured on each path

        self.X_hist.append(X)
        self.W_hist.append(W)
        self.pi_hist.append(pi)
        self.pi_plus_hist.append(pi * 0.0)

        self.WT = W
        self.piT = pi

        W = W * tf.cast(W_pos, tf.float64) # if bankrupcy occured on a path, set its termnial wealths to be 0
        W = tf.maximum(W,0.0001) # if terminal wealths is not positive, set it to a very small positive number to calculate the terminal utility

        self.loss = tf.reduce_mean(self.Opt_value - self.Utility(W * (1- np.abs(pi) * self.transaction_fee)))
        self.update_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'worker1')

    def simulate_change(self,Xj,Wj,pi_j,dZ): # Simulate dX and dW
        dt = self.T / self.N
        fct = 1 - exp(-self.lmda * dt)
        Ntdelta = dZ * self.sigma * sqrt((1-exp(-2*self.lmda*dt))/(2*self.lmda))
        dX = (self.mu - Xj) * fct + Ntdelta
        dW = pi_j * Wj * (-fct + self.mu * fct / Xj) + (1-pi_j) * Wj * (exp(self.r*dt)-1) + pi_j * Wj / Xj * Ntdelta
        return dX, dW



    def process_save_data(self, limit, x, w, p, p_plus, dx, lb, ub, make_plot=False): # Save data and make plots
        plt.ioff()
        x = np.array(x).reshape([self.N+1,self.num_path])
        w = np.array(w).reshape([self.N+1,self.num_path])
        p = np.array(p).reshape([self.N+1,self.num_path])
        p_plus = np.array(p_plus).reshape([self.N+1,self.num_path])
        dx = np.array(dx).reshape([self.N,self.num_path])
        lb = np.array(lb).reshape([self.N,self.num_path])
        ub = np.array(ub).reshape([self.N,self.num_path])
        np.save('result'+str(limit)+'.npy',{  'limit' : limit,
                                'x' : x,
                                'w' : w,
                                'p' : p,
                                'p_plus' : p_plus,
                                'dx' : dx,
                                'lb' : lb,
                                'ub' : ub})
        if make_plot:
            num = 0
            for j in np.linspace(0,self.num_path-1,20):
                i = int(j)
                fig = plt.figure(num,figsize=(16.0, 10.0))
                plt.subplot(211)
                plt.plot(w[:,i])
                plt.title('Cumulative wealth')
                ax1 = fig.add_subplot(212)
                init_level, = ax1.plot(np.ones([self.N+1]) * self.mu,'tab:green')
                x_path, = ax1.plot(x[:,i],'tab:blue')
                ax2 = ax1.twinx()
                position, = ax2.plot(p_plus[:,i],'tab:orange')
                ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
                plt.legend([init_level,x_path,position],['mean level','Asset price', 'position'])
                plt.title('Asset price and position')
                plt.savefig('limit-'+str(limit)+'-path'+str(i)+'.png')
                plt.close()
                num += 1

            plt.figure(num, figsize=(16.0, 10.0))
            subplot_count = 1
            for j in np.linspace(10,self.N-1,4):
                i = int(j)
                plt.subplot(2,2,subplot_count)
                ind = x[i].argsort()
                low_bound, = plt.plot(x[i][ind],lb[i][ind],'tab:blue')
                up_bound, = plt.plot(x[i][ind],ub[i][ind],'tab:orange')
                zero_pos, = plt.plot(x[i][ind],lb[i][ind] * 0, 'g--')
                plt.ylim(-8,30)
                plt.xlabel('asset price')
                plt.ylabel('proportion to invest')
                plt.legend([low_bound,up_bound,zero_pos],['lb', 'ub','pi=0'])
                plt.title('Trading zone at t=%.2f'%(i/self.N))
                subplot_count += 1
            plt.savefig('limit-' + str(limit) + '-trading-zones.png')
            plt.close()
            num += 1


    def train(self):
        sess = self.sess
        trainable_variables = tf.trainable_variables()
        self.global_step = tf.get_variable('global_step', [], initializer=const_init(1), trainable=False,
                                           dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(1.0, self.global_step, decay_steps=500, decay_rate=0.5,staircase=False)
        grads = tf.gradients(self.loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer( learning_rate=learning_rate  )
        apply_op = optimizer.apply_gradients( zip(grads, trainable_variables), global_step=self.global_step, name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        train_op = tf.group(*train_ops)
        multiplier = 10
        sess.run(tf.global_variables_initializer())
        limit = self.limit

        sess.run(self.global_step.initializer)
        self.num_path = 100 * multiplier
        for i in range(self.n_steps):
            sess.run(train_op,feed_dict={
                    self.X : np.ones([self.num_path,1]) * self.Xinit,
                    self.W : np.ones([self.num_path,1]) * self.Winit,
                    self.pi : np.zeros([self.num_path,1]),
                    self.dZ : np.random.normal(size=[self.num_path,self.N]),
                    self.W_pos: np.ones([self.num_path,1],dtype=bool),
                    self.upper_limit : np.ones([1,1],dtype=float) * limit,
                    self.lower_limit : np.ones([1,1],dtype=float) * (-limit),
                    self.is_training : True})

            if (i % 50 == 0) or (i == self.n_steps - 1):
                self.num_path = 10000
                step, loss, X_hist, W_hist, pi_hist, pi_plus_hist, dX_hist, \
                    lower_bound_hist, upper_bound_hist = sess.run([self.global_step,self.loss, \
                                                                   self.X_hist, self.W_hist, self.pi_hist, self.pi_plus_hist, \
                                                                   self.dX_hist, self.lower_bound_hist, self.upper_bound_hist],feed_dict={
                        self.X : np.ones([self.num_path,1]) * self.Xinit,
                        self.W : np.ones([self.num_path,1]) * self.Winit,
                        self.pi : np.zeros([self.num_path,1]),
                        self.dZ: np.random.normal(size=[self.num_path, self.N]),
                        self.W_pos: np.ones([self.num_path, 1], dtype=bool),
                    self.upper_limit: np.ones([1, 1], dtype=float) * limit,
                    self.lower_limit: np.ones([1, 1], dtype=float) * (-limit),
                        self.is_training : False})
                self.process_save_data(limit,X_hist,W_hist,pi_hist,pi_plus_hist,dX_hist,lower_bound_hist,upper_bound_hist, make_plot=(i == self.n_steps - 1))
                print('limit = %f, step = %d, loss = %f' %(limit,step, loss))
                self.num_path = 100 * multiplier

with tf.Session() as sess:
    tf.set_random_seed(0)
    np.random.seed(0)
    model = worker(sess)
    model.Play_Opt_Policy()
    model.build()
    model.train()
