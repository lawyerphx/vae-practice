import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data

sns.set_style(style='whitegrid')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def inference_network(x,latent_dim, hidden_size):   #encoder
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.flatten(x)
        net = slim.fully_connected(net,hidden_size)
        net = slim.fully_connected(net,hidden_size)
        params = slim.fully_connected(net, 2*latent_dim, activation_fn=None)
    loc = params[:,:latent_dim]
    scale = tf.nn.softplus(params[: ,latent_dim:])
    return loc,scale

def generative_network(z, hidden_size): #back
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.fully_connected(z, hidden_size)
        net = slim.fully_connected(net, hidden_size)
        Bernoulli_logits =slim.fully_connected(net,784,activation_fn=None)
        Bernoulli_logits = tf.reshape(Bernoulli_logits, [-1,28,28,1])
    return tf.nn.softplus(Bernoulli_logits)

LATENT = 10
n_sample = 1

# elbo_i = KL(p(z) || q(z|x_i)) + E_{z \~  q(z|x_i)}[p(x_i|z)]
def orgin(x,p_z,q_z_given_x,p_x_given_z):
    kl = tf.reduce_sum(tf.distributions.kl_divergence(p_z,q_z_given_x))
    expected_log = tf.reduce_sum(p_x_given_z.log_prob(x))
    return tf.reduce_sum(expected_log - kl)

#def SGVB():

#def

#def vimco():

def main():
    x = tf.placeholder(tf.float32, [None, 28,28,1])
    q_mu, q_sigma = inference_network(x,LATENT,100)
    q_z_given_x = tf.distributions.Normal(loc=q_mu, scale=q_sigma)
    p_x_given_z_logits = generative_network(q_z_given_x.sample(), 100)
    p_x_given_z = tf.distributions.Bernoulli(logits=p_x_given_z_logits)

    p_z = tf.distributions.Normal(loc = np.zeros(LATENT,np.float32), scale= np.ones(LATENT,np.float32))

    elbo = orgin(x,p_z,q_z_given_x,p_x_given_z)

    solve = tf.train.RMSPropOptimizer(0.001).minimize(-elbo)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        np_x,_  = mnist.train.next_batch(20)
        np_x = np_x.reshape(20,28,28,1)
        sess.run(solve, feed_dict={x:np_x})
#        print("kl = " ,sess.run(kl, feed_dict={x:np_x}))
#        print("sam = ",sess.run(samp_x_given_z, feed_dict={x:np_x}))
#        print("exp = ",sess.run(expected_log_likelihood, feed_dict={x:np_x}))
        print(-sess.run(elbo, feed_dict={x:np_x}))

if __name__ == '__main__':
    main()
