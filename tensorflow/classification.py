import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense
from spektral.data.loaders import BatchLoader
from spektral.datasets import TUDataset
from spektral.layers import GraphMasking
from spektral.layers.pooling import GlobalAvgPool
from GTVConv import GTVConv
from AsymCheegerCutPool import AsymCheegerCutPool


################################
# CONFIG
################################
mp_layers = 1
mp_channels = 32
mp_activation = "relu"
delta_coeff = 2.0

mlp_hidden_layers = 1
mlp_hidden_channels = 32
mlp_activation = "relu"
totvar_coeff=0.5
balance_coeff=0.5

batch_size = 16
l2_reg_val = 0
learning_rate = 5e-4
epochs = 100
patience = 10


################################
# LOAD DATASET
################################
dataset = TUDataset("PROTEINS", clean=True)

# Parameters
N = max(g.n_nodes for g in dataset)
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr = dataset[idx_tr]
dataset_va = dataset[idx_va]
dataset_te = dataset[idx_te]
loader_tr = BatchLoader(dataset_tr, batch_size=batch_size, mask=True)
loader_va = BatchLoader(dataset_va, batch_size=batch_size, shuffle=False, mask=True)
loader_te = BatchLoader(dataset_te, batch_size=batch_size, shuffle=False, mask=True)


################################
# MODEL
################################
class ClassificationModel(Model):
    def __init__(self, n_out, mp1, pool1, mp2, pool2, mp3):
        super().__init__()
        
        self.mask = GraphMasking()
        self.mp1 = mp1
        self.pool1 = pool1
        self.mp2 = mp2
        self.pool2 = pool2
        self.mp3 = mp3
        self.global_pool = GlobalAvgPool()
        self.output_layer = Dense(n_out, activation = "softmax")
        
    def call(self, inputs):
        
        x, a = inputs
        x = self.mask(x)
        out = x
        
        # 1st block
        for _mp in self.mp1:
            out = _mp([out, a])
        out, a_pool = self.pool1([out, a])
        
        # 2nd block
        for _mp in self.mp2:
            out = _mp([out, a_pool])
        out, a_pool = self.pool2([out, a_pool])
        
        # 3rd block
        for _mp in self.mp3:
            out = _mp([out, a_pool])
        out = self.global_pool(out)
        out = self.output_layer(out)
        
        return out


MP1 = [GTVConv(mp_channels, 
               delta_coeff=delta_coeff,
               activation=mp_activation,
               kernel_regularizer=L2(l2_reg_val))
       for _ in range(mp_layers)]

Pool1 = AsymCheegerCutPool(int(N//2), 
                           mlp_hidden=[mlp_hidden_channels
                                        for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           kernel_regularizer=L2(l2_reg_val))

MP2 = [GTVConv(mp_channels, 
               delta_coeff=delta_coeff,
               activation=mp_activation,
               kernel_regularizer=L2(l2_reg_val))
       for _ in range(mp_layers)]

Pool2 = AsymCheegerCutPool(int(N//4), 
                           mlp_hidden=[mlp_hidden_channels
                                        for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           kernel_regularizer=L2(l2_reg_val))

MP3 = [GTVConv(mp_channels, 
               delta_coeff=delta_coeff,
               activation=mp_activation,
               kernel_regularizer=L2(l2_reg_val))
       for _ in range(mp_layers)]
                                                       

# Compile the model
model = ClassificationModel(
    n_out, 
    mp1=MP1, 
    pool1=Pool1, 
    mp2=MP2, 
    pool2=Pool2, 
    mp3=MP3)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["acc"])
            
################################
# TRAIN AND TEST
################################

model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    epochs=epochs,
    validation_data=loader_va,
    validation_steps=loader_va.steps_per_epoch,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
    verbose = 2)

loss_te, acc_te = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Test loss: {}. Test acc: {}".format(loss_te, acc_te))
