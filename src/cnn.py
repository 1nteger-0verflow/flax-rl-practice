import jax
from flax import nnx


class CNN(nnx.Module):
    def __init__(self, dropout_rate: float = 0.025, *, rngs: nnx.Rngs):
        dim1 = 32
        dim2 = 64
        size2 = (7, 7)
        dim3 = 256
        out_dim = 10

        self.conv1 = nnx.Conv(in_features=1, out_features=dim1, kernel_size=(3, 3), rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(num_features=dim1, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate)
        self.conv2 = nnx.Conv(in_features=dim1, out_features=dim2, kernel_size=(3, 3), rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(num_features=dim2, rngs=rngs)
        self.linear1 = nnx.Linear(in_features=dim2 * size2[0] * size2[1], out_features=dim3, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate)
        self.linear2 = nnx.Linear(in_features=dim3, out_features=out_dim, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs | None = None):
        x = nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs)))
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.batch_norm2(self.conv2(x)))
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)  # flatten (7, 7, dim2) -> (7*7*dim2, )
        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        return self.linear2(x)


rngs = nnx.Rngs(0)
model = CNN(rngs=rngs)

nnx.display(model)
