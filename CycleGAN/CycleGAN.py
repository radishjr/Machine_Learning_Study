
"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 15             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = N_IDEAS     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
print("np.linspace(-1, 1, ART_COMPONENTS)")
print(np.linspace(-1, 1, ART_COMPONENTS))
print("PAINT_POINTS")
print(PAINT_POINTS)

# show our beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='x upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#74BCFF', lw=3, label='x lower bound')

plt.plot(PAINT_POINTS[0], -(2 * np.power(PAINT_POINTS[0], 2) + 1), c='#FF9359', lw=3, label='y upper bound')
plt.plot(PAINT_POINTS[0], -(1 * np.power(PAINT_POINTS[0], 2) + 0), c='#FF9359', lw=3, label='y lower bound')
plt.legend(loc='upper right')
plt.show()

def new_artist_work():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    return paintings

with tf.variable_scope('Generator_X_To_Y'):
    real_art_X = tf.placeholder(tf.float32, [None, N_IDEAS])          # random ideas (could from normal distribution)
    G_l1_X = tf.layers.dense(real_art_X, 128, tf.nn.relu)
    G_out_Y = tf.layers.dense(G_l1_X, ART_COMPONENTS)               # making a painting from these random ideas

print("real_art_X")
print(real_art_X)
print("G_out_Y")
print(G_out_Y)


with tf.variable_scope('Discriminator_Y'):
    real_art_Y = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in_y')   # receive art work from the famous artist
    D_l0_Y = tf.layers.dense(real_art_Y, 128, tf.nn.relu, name='l')
    prob_artist0_Y = tf.layers.dense(D_l0_Y, 1, tf.nn.sigmoid, name='out_y')              # probability that the art work is made by artist
    # reuse layers for generator
    D_l1 = tf.layers.dense(G_out_Y, 128, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
    prob_artist1_Y = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out_y', reuse=True)  # probability that the art work is made by artist

##### MLE Loss Function ####
D_loss_Y = -tf.reduce_mean(tf.log(prob_artist0_Y) + tf.log(1-prob_artist1_Y))
G_loss_Y = tf.reduce_mean(tf.log(1-prob_artist1_Y))


with tf.variable_scope('Generator_Y_To_X'):
    G_in_Y = G_out_Y          # random ideas (could from normal distribution)
    G_l1_Y = tf.layers.dense(G_in_Y, 128, tf.nn.relu)
    G_out_X = tf.layers.dense(G_l1_Y, ART_COMPONENTS)               # making a painting from these random ideas

with tf.variable_scope('Discriminator_X'):
    D_l0_X = tf.layers.dense(real_art_X, 128, tf.nn.relu, name='l')
    prob_artist0_X = tf.layers.dense(D_l0_X, 1, tf.nn.sigmoid, name='out_y')              # probability that the art work is made by artist
    # reuse layers for generator
    D_l1_X = tf.layers.dense(G_out_X, 128, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G
    prob_artist1_X = tf.layers.dense(D_l1_X, 1, tf.nn.sigmoid, name='out_y', reuse=True)  # probability that the art work is made by artist

##### MLE Loss Function ####
D_loss_X = -tf.reduce_mean(tf.log(prob_artist0_X) + tf.log(1-prob_artist1_X))
G_loss_X = tf.reduce_mean(tf.log(1-prob_artist1_X))

train_D_Y = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss_Y, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G_Y = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss_Y, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

train_D_X = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss_X, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G_X = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss_X, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about continuous plotting
for step in range(50000):
    artist_paintings_x = new_artist_work()           # real painting from artist
    artist_paintings_y = -new_artist_work()           # real painting from artist

    G_paintings_Y, G_paintings_X, pa0_y, pa0_x, Dl_y, Dl_x = sess.run([G_out_Y, 
                                    G_out_X, 
                                    prob_artist0_Y, 
                                    prob_artist0_X, 
                                    D_loss_Y, 
                                    D_loss_X, 
                                    train_D_X, train_G_X, 
                                    train_D_Y, train_G_Y],    # train and get results
                                    {
                                        real_art_X: artist_paintings_x,
                                        real_art_Y: artist_paintings_y,
                                    })[:6]

    if step % 50 == 0:  # plotting
        plt.cla()

        plt.plot(PAINT_POINTS[0], artist_paintings_x[0], c='#4A6631', lw=3, label='real painting X',)
        plt.plot(PAINT_POINTS[0], artist_paintings_y[0], c='#992A21', lw=3, label='real painting Y',)
        

        plt.plot(PAINT_POINTS[0], G_paintings_X[0], c='#4AD631', lw=3, label='Generated painting X',)
        plt.plot(PAINT_POINTS[0], G_paintings_Y[0], c='#FF5A51', lw=3, label='Generated painting Y',)
        
        plt.plot(PAINT_POINTS[0], (2 * np.power(PAINT_POINTS[0], 2) + 1), c='#74BCFF', lw=3, label='x upper bound')
        plt.plot(PAINT_POINTS[0], (1 * np.power(PAINT_POINTS[0], 2) + 0), c='#74BCFF', lw=3, label='x lower bound')

        plt.plot(PAINT_POINTS[0], -(2 * np.power(PAINT_POINTS[0], 2) + 1), c='#FF9359', lw=3, label='y upper bound')
        plt.plot(PAINT_POINTS[0], -(1 * np.power(PAINT_POINTS[0], 2) + 0), c='#FF9359', lw=3, label='y lower bound')
        
        plt.text(-.5, 2.3, 'D_y accuracy=%.2f (0.5 for D to converge), score= %.2f (-1.38 for G to converge)' % (pa0_y.mean(), -Dl_y), fontdict={'size': 15})
        plt.text(-.5, 2, 'D_x accuracy=%.2f (0.5 for D to converge), score= %.2f (-1.38 for G to converge)' % (pa0_x.mean(), Dl_x), fontdict={'size': 15})
        
        plt.ylim((-3, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()