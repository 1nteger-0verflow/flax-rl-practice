import jax
import jax.numpy as jnp


def selu(x: jax.Array, alpha: float = 1.67, lambda_: float = 1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)


@jax.jit
def sum_logistic(x: jax.Array):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))


sum_logistic_1 = jax.grad(sum_logistic)

x = jnp.arange(5.0)
y = selu(x)
z = sum_logistic_1(x)
print(y)
