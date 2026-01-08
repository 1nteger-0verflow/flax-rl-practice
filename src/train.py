import pickle
import random
from collections import deque
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import lz4.frame as lz4f
import optax
import orbax.checkpoint as ocp
from flax import nnx

from modules.model import DQNCNN
from modules.wrapper import get_atari_env


class _Data(TypedDict):
    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_states: jax.Array
    dones: jax.Array


@nnx.jit
def train_step(
    online_network: nnx.Module, target_network: nnx.Module, data: _Data, optimizer: nnx.Optimizer, gamma: float = 0.997
):
    def loss_fn(online_network: nnx.Module):
        q_values = online_network(data["states"])
        q_values_selected = q_values[jnp.arange(len(data["actions"])), data["actions"]]

        next_q_values = target_network(data["next_states"])
        max_next_q_values = jnp.max(next_q_values, axis=1)
        targets = data["rewards"] + gamma * max_next_q_values * (1 - data["dones"])

        # MSE instead of Huber loss for simplicity
        return jnp.mean((q_values_selected - targets) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(online_network)
    optimizer.update(online_network, grads)

    return loss.mean()


class ReplayBuffer:
    def __init__(self, maxlen: int, *, rngs: nnx.Rngs):
        self.buffer: deque[bytes | bytearray] = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self._rngs = rngs

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done: int):
        data = (state, action, reward, next_state, done)
        self.buffer.append(lz4f.compress(pickle.dumps(data)))

    def sample_batch(self, batch_size: int):
        indices = self._rngs.permutation(len(self.buffer))[:batch_size]
        samples = [pickle.loads(lz4f.decompress(self.buffer[i])) for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return _Data(
            states=jnp.array(states),
            actions=jnp.array(actions),
            rewards=jnp.array(rewards),
            next_states=jnp.array(next_states),
            dones=jnp.array(dones),
        )


def main():
    outdir = Path("./out")
    rngs = nnx.Rngs(0)
    env = get_atari_env("Breakout-v4", record_folder=outdir / "mp4", record_frequency=100)
    action_dim = int(env.action_space.n)

    online_network = DQNCNN(in_dim=4, out_dim=action_dim, rngs=rngs)
    target_network = DQNCNN(in_dim=4, out_dim=action_dim, rngs=rngs)
    optimizer = nnx.Optimizer(model=online_network, tx=optax.adam(learning_rate=2e-4), wrt=nnx.Param)

    replay_buffer = ReplayBuffer(maxlen=250_000, rngs=rngs)

    global_steps, global_episodes = 0, 0
    while global_steps < 2_000_000:
        state, info = env.reset()
        ep_rewards, ep_steps = 0, 0
        lives = info["lives"]

        while True:
            epsilon = max(0.1, 1.0 - 0.9 * global_steps / 100_000)
            if epsilon > rngs.uniform():
                # Random action (exploration)
                action = env.action_space.sample()
            else:
                # Greedy action (exploitation)
                qvalues = online_network(jnp.expand_dims(state, axis=0))
                action = jnp.argmax(qvalues, axis=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # life loss as episode end
            life_loss = lives != info["lives"]
            lives = info["lives"]

            replay_buffer.add(state, action, jnp.clip(reward, -1, 1), next_state, int(life_loss))

            # Update network
            if len(replay_buffer) > 1000 and global_steps % 4 == 0:
                batch_data = replay_buffer.sample_batch(64)
                _loss = train_step(online_network, target_network, batch_data, optimizer)

            # Sync target network
            if global_steps % 10_000 == 0:
                """Copy weights from online network to target network."""
                _graphdef, _state = nnx.split(online_network)
                nnx.update(target_network, _state)

            # Save model checkpoint
            if global_steps % 100_000 == 0:
                checkpointer = ocp.StandardCheckpointer()
                _graphdef, _state = nnx.split(online_network)
                checkpointer.save(outdir / f"ckpt_{global_steps}", _state)

            state = next_state
            ep_rewards += reward
            ep_steps += 1
            global_steps += 1

            if done:
                print("====" * 5)
                print(f"Episode {global_episodes} finished after {ep_steps} steps with reward {ep_rewards}")
                print(f"Global step: {global_steps}")

                global_episodes += 1
                break

    env.close()
