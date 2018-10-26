import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

LATENT = 2
n_sample = 1
INPUT_SIZE = 28*28
BATCH_SIZE = 10
HIDDEN_SIZE = 100

#sns.set_style(style='whitegrid')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(data, int_size, out_size,act_fn = None):
    Weights = tf.Variable(tf.random_normal([int_size, out_size],0,1))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    template = tf.matmul(data, Weights) + biases
    if act_fn == None:
        return template
    else:
        return act_fn(template)

def inference_network(x, input_size, latent_dim, hidden_size):   #encoder
    net = add_layer(x, input_size,hidden_size,act_fn=tf.nn.relu)
    net = add_layer(net, hidden_size,hidden_size,act_fn=tf.nn.relu)
    params = add_layer(net,hidden_size, 2*latent_dim)
    loc = params[:, :latent_dim]
    scale = tf.nn.softplus(params[: ,latent_dim:]) + 1e-6
    return loc, scale

def generative_network(x, input_size, hidden_size): #back
    net = add_layer(x, input_size, hidden_size, act_fn=tf.nn.relu)
    net = add_layer(net, hidden_size, hidden_size, act_fn=tf.nn.relu)
    Bernoulli_logits = add_layer(net,hidden_size,INPUT_SIZE)
    return Bernoulli_logits

# elbo_i = E_{z \~  q(z|x_i)}[p(x_i|z)] - KL(p(z) || q(z|x_i))

def main():
    ds = tf.distributions
    x = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_SIZE])  #[Batch_size, input_size]
    q_mu, q_sigma = inference_network(x, INPUT_SIZE, LATENT, HIDDEN_SIZE) #q_mu [None, 100], q_sigma [None, 100]
    q_z_given_x = ds.Normal(loc=q_mu, scale=q_sigma)  #100个高斯分布
    p_x_given_z_logits = generative_network(q_z_given_x.sample(), LATENT, HIDDEN_SIZE)
    p_x_given_z = ds.Bernoulli(logits=p_x_given_z_logits)

    p_z = ds.Normal(loc = np.zeros(LATENT,np.float32), scale= np.ones(LATENT,np.float32))
#    print(p_z)
    kl = tf.reduce_sum(ds.kl_divergence(p_z, q_z_given_x),1)
    marginal_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), 1)
    elbo = tf.reduce_sum(marginal_likelihood - kl)
    loss = -elbo
    solve = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        np_x, npx_y  = mnist.train.next_batch(BATCH_SIZE)
        #print(sess.run([q_mu,q_sigma,kl],feed_dict={x:np_x}))
        sess.run(solve, feed_dict={x:np_x})
        print(sess.run(loss, feed_dict={x:np_x}))
    sess.close()

if __name__ == '__main__':
    main()
