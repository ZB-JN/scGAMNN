from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE, KLD
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from spektral.layers import GATConv, TAGConv,GCNConv
from tensorflow.keras.initializers import GlorotUniform
from memory_profiler import profile
from layers import *
import tensorflow_probability as tfp
from utils import *

LR_DECAY_FACTOR = 0.8
DECAY_STEP_SIZE = 25
MIN_LR = 1e-6

# Ref:  https://github.com/zoubin-ai/deepMNN
def update_lr(optimizer, epoch, init_lr, min_lr=MIN_LR, decay_step_size=DECAY_STEP_SIZE, lr_decay_factor=LR_DECAY_FACTOR):
    """ stepwise learning rate calculator """
    exponent = int(np.floor((epoch + 1) / decay_step_size))
    lr = init_lr * np.power(lr_decay_factor, exponent)
    if lr < min_lr:
        optimizer.learning_rate = min_lr
    else:
        optimizer.learning_rate = lr
    print('Learning rate = %.7f' % optimizer.learning_rate)
    # return lr

# Ref: https://github.com/ZixiangLuo1161/scGAE
# Modification made to scGAE with MIT License
class scGAMNN(tf.keras.Model):

    def __init__(self, X, adj, adj_n,match,hidden_dim=120, latent_dim=20, dec_dim=None, adj_dim=64,
                 decA="DBL", layer_enc="GAT"):
        super(scGAMNN, self).__init__()
        if dec_dim is None:
            dec_dim = [64, 256, 512]
        self.latent_dim = latent_dim
        self.X = np.float64(X)
        self.adj = np.float32(adj)
        self.adj_n = np.float32(adj_n)
        self.match = match
        self.n_sample = X.shape[0]
        self.in_dim = X.shape[1]
        self.sparse = False

        initializer = GlorotUniform(seed=7)

        # Encoder
        X_input = Input(shape=self.in_dim)
        h = Dropout(0.2)(X_input)
        if layer_enc == "GAT":
            A_in = Input(shape=self.n_sample)
            h = GATConv(channels=hidden_dim, attn_heads=1, kernel_initializer=initializer, activation="relu")([h, A_in])
            z_mean = GATConv(channels=latent_dim, kernel_initializer=initializer, attn_heads=1)([h, A_in])
        elif layer_enc == "GCN":
            A_in = Input(shape=self.n_sample)
            h = GCNConv(channels=hidden_dim,  kernel_initializer=initializer, activation="relu")([h, A_in])
            z_mean = GCNConv(channels=latent_dim, kernel_initializer=initializer)([h, A_in])
        elif layer_enc == "TAG":
            self.sparse = True
            A_in = Input(shape=self.n_sample, sparse=True)
            h = TAGConv(channels=hidden_dim, kernel_initializer=initializer, activation="relu")([h, A_in])
            z_mean = TAGConv(channels=latent_dim, kernel_initializer=initializer)([h, A_in])
        # else:
        #     print('None')

        self.encoder = Model(inputs=[X_input, A_in], outputs=z_mean, name="encoder")
        clustering_layer = ClusteringLayer(name='clustering')(z_mean)
        self.cluster_model = Model(inputs=[X_input, A_in], outputs=clustering_layer, name="cluster_encoder")

        # Adjacency matrix decoder
        if decA == "DBL":
            dec_in = Input(shape=latent_dim)
            h = Dense(units=adj_dim, activation=None)(dec_in)
            h = Bilinear()(h)
            dec_out = Lambda(lambda z: tf.nn.sigmoid(z))(h)
            self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoder1")
        elif decA == "BL":
            dec_in = Input(shape=latent_dim)
            h = Bilinear()(dec_in)
            dec_out = Lambda(lambda z: tf.nn.sigmoid(z))(h)
            self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoder1")
        elif decA == "IP":
            dec_in = Input(shape=latent_dim)
            dec_out = Lambda(lambda z: tf.nn.sigmoid(tf.matmul(z, tf.transpose(z))))(dec_in)
            self.decoderA = Model(inputs=dec_in, outputs=dec_out, name="decoder1")
        else:
            self.decoderA = None

        # Expression matrix decoder
        decx_in = Input(shape=latent_dim)
        h = Dense(units=dec_dim[0], activation="relu")(decx_in)
        h = Dense(units=dec_dim[1], activation="relu")(h)
        h = Dense(units=dec_dim[2], activation="relu")(h)
        decx_out = Dense(units=self.in_dim)(h)
        self.decoderX = Model(inputs=decx_in, outputs=decx_out, name="decoderX")

    # @profile
    def train(self,epochs=80, min_epochs=30,early_stopping=10,decay_step_size=DECAY_STEP_SIZE, lr=2e-3, W_a=1, W_x=1,W_w=0):

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if self.sparse == True:
            self.adj_n = tfp.math.dense_to_sparse(self.adj_n)

        # Training
        losses=[]
        for epoch in range(1, epochs + 1):
            with tf.GradientTape(persistent=True) as tape:
                z = self.encoder([self.X, self.adj_n])
                X_out = self.decoderX(z)
                A_out = self.decoderA(z)
                A_rec_loss = tf.reduce_mean(MSE(self.adj, A_out))
                X_rec_loss = tf.reduce_mean(MSE(self.X, X_out))

                MNN_loss = 0
                z_gather1 = tf.gather(z, axis=0, indices=self.match[:, 0])
                z_gather2 = tf.gather(z, axis=0, indices=self.match[:, 1])
                MNN_loss += tf.reduce_sum(tf.norm(z_gather1 - z_gather2,ord=2,axis=1)) / len(self.match)
                loss = W_a * A_rec_loss + W_x * X_rec_loss + W_w * MNN_loss


            losses.append(loss.numpy())

            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

            print("Epoch", epoch, " X_rec_loss:", X_rec_loss.numpy(), "  A_rec_loss:", A_rec_loss.numpy(),
                   "  MNN_loss:", A_rec_loss.numpy()," total_loss: ", loss.numpy())

            update_lr(optimizer, epoch, init_lr=lr, decay_step_size=decay_step_size)

            if epoch > min_epochs and losses[-1] > np.mean(losses[-(early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("train Finish!")


    def embedding(self, count, adj_n):
        if self.sparse:
            adj_n = tfp.math.dense_to_sparse(adj_n)
        return np.array(self.encoder([count, adj_n]))

    def decoder(self, count, adj_n):
        if self.sparse:
            adj_n = tfp.math.dense_to_sparse(adj_n)
        z=self.encoder([count, adj_n])
        return np.array(self.decoderX(z))

    def rec_A(self, count, adj_n):
        h = self.encoder([count, adj_n])
        rec_A = self.decoderA(h)
        return np.array(rec_A)


