import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from spektral.layers import ops
from spektral.layers.pooling.src import SRCPool

class AsymCheegerCutPool(SRCPool):
    """
    Placeholder
    """

    def __init__(self,
                 k,
                 mlp_hidden=None,
                 mlp_activation="relu",
                 return_selection=False,
                 use_bias=True,
                 totvar_coeff=1.0,
                 balance_coeff=1.0,
                 softmax_temperature=1.0,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                 ):
        super().__init__(
            k=k,
            mlp_hidden=mlp_hidden,
            mlp_activation=mlp_activation,
            return_selection=return_selection,
            use_bias=use_bias,
            softmax_temperature=softmax_temperature,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.k = k
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = mlp_activation
        self.totvar_coeff = totvar_coeff
        self.balance_coeff = balance_coeff

    def build(self, input_shape):
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )
        self.mlp = Sequential(
            [
                Dense(channels, self.mlp_activation, **layer_kwargs)
                for channels in self.mlp_hidden
            ]
            + [Dense(self.k, "softmax", **layer_kwargs)]
        )

        super().build(input_shape)

    def call(self, inputs, mask=None):
        x, a, i = self.get_inputs(inputs)
        return self.pool(x, a, i, mask=mask)

    def select(self, x, a, i, mask=None):
        s = self.mlp(x)
        if mask is not None:
            s *= mask[0]

        # Total variation loss
        cut_loss = self.totvar_loss(a, s)
        if K.ndim(a) == 3:
            cut_loss = K.mean(cut_loss)
        self.add_loss(self.totvar_coeff * cut_loss)

        # Asymmetric l1-norm loss
        bal_loss = self.balance_loss(s)
        if K.ndim(a) == 3:
            bal_loss = K.mean(bal_loss)
        self.add_loss(self.balance_coeff * bal_loss)

        return s

    def reduce(self, x, s, **kwargs):
        return ops.modal_dot(s, x, transpose_a=True)

    def connect(self, a, s, **kwargs):
        a_pool = ops.matmul_at_b_a(s, a)
        
        return a_pool

    def reduce_index(self, i, s, **kwargs):
        i_mean = tf.math.segment_mean(i, i)
        i_pool = ops.repeat(i_mean, tf.ones_like(i_mean) * self.k)

        return i_pool
    
    def totvar_loss(self, a, s):
        if K.is_sparse(a):
            index_i = a.indices[:, 0]
            index_j = a.indices[:, 1]

            n_edges = float(len(a.values))

            loss = tf.math.reduce_sum(a.values[:, tf.newaxis] * 
                                      tf.math.abs(tf.gather(s, index_i) -
                                                  tf.gather(s, index_j)),
                                      axis=(-2, -1))

        else:
            n_edges = tf.cast(tf.math.count_nonzero(
                a, axis=(-2, -1)), dtype=s.dtype)
            n_nodes = tf.shape(a)[-1]
            if K.ndim(a) == 3:
                loss = tf.math.reduce_sum(a * tf.math.reduce_sum(tf.math.abs(s[:, tf.newaxis, ...] -
                                                                             tf.repeat(s[..., tf.newaxis, :],
                                                                                       n_nodes, axis=-2)), axis=-1),
                                          axis=(-2, -1))
            else:
                loss = tf.math.reduce_sum(a * tf.math.reduce_sum(tf.math.abs(s -
                                                                             tf.repeat(s[..., tf.newaxis, :],
                                                                                       n_nodes, axis=-2)), axis=-1),
                                          axis=(-2, -1))

        loss *= 1 / (2 * n_edges)

        return loss

    def balance_loss(self, s):
        n_nodes = tf.cast(tf.shape(s, out_type=tf.int32)[-2], s.dtype)

        # k-quantile
        idx = tf.cast(tf.math.floor(n_nodes / self.k) + 1, dtype=tf.int32)
        med = tf.math.top_k(tf.linalg.matrix_transpose(s), # Note: Could probably do this without topk
                            k=idx).values[..., -1]
        # Asymmetric l1-norm
        if K.ndim(s) == 2:
            loss = s - med
        else:
            loss = s - med[:, tf.newaxis, ...]
        loss = ((tf.cast(loss >= 0, loss.dtype) * (self.k - 1) * loss) +
                (tf.cast(loss < 0, loss.dtype) * loss * -1.))
        loss = tf.math.reduce_sum(loss, axis=(-2, -1))
        loss = 1 / (n_nodes * (self.k - 1)) * (n_nodes * (self.k - 1) - loss)

        return loss

    def get_config(self):
        config = {
            "k": self.k,
            "mlp_hidden": self.mlp_hidden,
            "mlp_activation": self.mlp_activation,
            "totvar_coeff": self.totvar_coeff,
            "balance_coeff": self.balance_coeff
        }
        base_config = super().get_config()
        return {**base_config, **config}
