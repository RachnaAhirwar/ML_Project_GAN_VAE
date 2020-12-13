import tensorflow as tf
import numpy as npy
from tensorflow import keras as krs
from tensorflow.keras import layers as lyr
import matplotlib.pyplot as plot

class VariationalAutoEncoder(krs.Model):

    def train_step(self, inp):
        if isinstance(inp, tuple):
            inp = inp[0]
            
        with tf.GradientTape() as gt:
            # mu and variance
            mu, var, z = vae_enc(inp)
            
            # re-construction
            recons = vae_dec(z)
            
            # KL loss
            loss_kl = tf.reduce_mean(1 - tf.exp(var) - tf.square(mu) + var) * -0.5
            
            # reconstruction loss
            loss_recons = tf.reduce_mean(krs.losses.binary_crossentropy(inp, recons)) * 28 * 28
            
            # loss total
            loss_total = loss_kl + loss_recons
            
        # gradients
        g = gt.gradient(loss_total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(g, self.trainable_weights))
        
        return {
            "kl loss is": loss_kl,
            "reconstruction loss is": loss_recons,
            "total loss is": loss_total,  
        }
    
    def __init__(self, vae_enc, vae_dec, **kwargs):
        # vae_enc -> vae encoder, vae_dec -> vae decoder
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.vae_dec = vae_dec
        self.vae_enc = vae_enc
        
    
class Sample(lyr.Layer):

    def plot_latent(vae_enc, vae_dec):
        # output axa images
        a = 30
        
        # size of digit
        dig_sz = 28
        
        # scale
        s = 2.0
        
        # figure size
        fig_sz = 15
        
        fig = npy.zeros((dig_sz * a, dig_sz * a))

        # linear scaling
        x = npy.linspace(-s, s, a)
        y = npy.linspace(-s, s, a)[::-1]

        # for all rows and columns
        for c, col in enumerate(y):
            for r, row in enumerate(x):
                
                smpl = npy.array([[row, col]])
                xd = vae_dec.predict(smpl)

                dig = xd[0].reshape(dig_sz, dig_sz)
                figure[
                    c * dig_sz : (c + 1) * dig_sz,
                    r * dig_sz : (r + 1) * dig_sz,
                ] = dig

        # plot figure
        plot.fig(fig_sz=(fig_sz, fig_sz))
        
        # start range
        st = dig_sz // 2
        
        # end range
        en = a * dig_sz + 1 + st
        pixel_range = npy.arange(st, en, dig_sz)
        
        # sample range, x and y
        x_range = npy.round(x, 1)
        y_range = npy.round(y, 1)
        
        # show plot
        plot.imshow(fig, cmap="Greys_r")
        plot.show()
        
    def call(self, inp):
        # input
        mu, var = inp
        
        # batch
        batch = tf.shape(mu)[0]
        
        #dim
        dim = tf.shape(mu)[1]
        
        ep = tf.keras.backend.random_normal(shape=(batch, dim))
        return ep * tf.exp(0.5 * var) + mu
    

#latent dim
dim_latent = 2

# inputs for encoder
inputs_encoder = krs.Input(shape=(28, 28, 1))

# -------------------encoder layers start--------------------
# convolution 2D
x = lyr.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs_encoder)
x = lyr.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)

# flatten
x = lyr.Flatten()(x)

# dense
x = lyr.Dense(16, activation="relu")(x)
mu = lyr.Dense(dim_latent, name="mu")(x)
var = lyr.Dense(dim_latent, name="var")(x)
z = Sample()([mu, var])
# --------------------encoder layers end----------------------

# encoder summary
vae_enc = krs.Model(inputs_encoder, [mu, var, z], name="vae_enc")
vae_enc.summary()


# latent inputs
inputs_latent = krs.Input(shape=(dim_latent,))
# ---------------------decoder layers start--------------------
# dense
x = lyr.Dense(7 * 7 * 64, activation="relu")(inputs_latent)

# reshape
x = lyr.Reshape((7, 7, 64))(x)

# convolution 2D transpose
x = lyr.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = lyr.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
outputs_decoder = lyr.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
# ---------------------decoder layers end ---------------------

# decoder summary
vae_dec = krs.Model(inputs_latent, outputs_decoder, name="vae_dec")
vae_dec.summary()


# load data
(training_x, _), (testing_x, _) = krs.datasets.mnist.load_data()
digits_dataset = npy.concatenate([training_x, testing_x], axis=0)
digits_dataset = npy.expand_dims(digits_dataset, -1).astype("float32") / 255

variationalAutoEncoder = VariationalAutoEncoder(vae_enc, vae_dec)
variationalAutoEncoder.compile(optimizer=krs.optimizers.Adam())
variationalAutoEncoder.fit(digits_dataset, epochs=18, batch_size=234)

# latent plot
plot_latent(vae_enc, vae_dec)
