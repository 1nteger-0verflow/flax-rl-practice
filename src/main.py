import jax


def main():
    print(f"JAX backend: {jax.default_backend()}")
    print(f"GPUs found: {jax.device_count()}")


if __name__ == "__main__":
    main()
