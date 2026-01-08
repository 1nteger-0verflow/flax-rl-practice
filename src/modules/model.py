import jax
import jax.numpy as jnp
from flax import nnx


class DQNCNN(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        dim1 = 32
        dim2 = 64
        dim3 = 64
        dim4 = 512

        self.conv1 = nnx.Conv(
            in_features=in_dim, out_features=dim1, kernel_size=(8, 8), strides=(4, 4), padding="VALID", rngs=rngs
        )
        self.norm1 = nnx.GroupNorm(num_features=dim1, num_groups=8, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=dim1, out_features=dim2, kernel_size=(4, 4), strides=(2, 2), padding="VALID", rngs=rngs
        )
        self.norm2 = nnx.GroupNorm(num_features=dim2, num_groups=8, rngs=rngs)
        self.conv3 = nnx.Conv(
            in_features=dim2, out_features=dim3, kernel_size=(3, 3), strides=(1, 1), padding="VALID", rngs=rngs
        )
        self.norm3 = nnx.GroupNorm(num_features=dim3, num_groups=8, rngs=rngs)
        self.fc1 = nnx.Linear(in_features=7 * 7 * dim3, out_features=dim4, rngs=rngs)
        self.head = nnx.Linear(in_features=dim4, out_features=out_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.relu(self.norm1(self.conv1(x)))
        x = nnx.relu(self.norm2(self.conv2(x)))
        x = nnx.relu(self.norm3(self.conv3(x)))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc1(x))
        return self.head(x)
