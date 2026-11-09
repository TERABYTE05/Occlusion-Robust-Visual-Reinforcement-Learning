from __future__ import annotations

import shutil

import numpy as np

_MAX_REJECTION_ROUNDS = 8

_COPY_CHUNK = 4096


class ReplayBuffer:

    def __init__(
        self,
        capacity: int,
        image_size: int = 84,
        frame_stack: int = 3,
        proprio_dim: int = 13,
        action_dim: int = 4,
        nstep: int = 1,
        discount: float = 0.99,
        seed: int | None = None,
    ):
        if capacity < frame_stack + 1:
            raise ValueError("capacity must exceed the frame stack")
        if nstep < 1:
            raise ValueError("nstep must be at least 1")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")

        self.capacity = int(capacity)
        self.image_size = int(image_size)
        self.frame_stack = int(frame_stack)
        self.nstep = int(nstep)
        self.discount = float(discount)

        self._frames = np.zeros((capacity, image_size, image_size, 3), dtype=np.uint8)
        self._proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=bool)
        self._truncated = np.zeros(capacity, dtype=bool)
        self._has_next = np.zeros(capacity, dtype=bool)
        self._age = np.zeros(capacity, dtype=np.int32)

        self._next = 0
        self._last: int | None = None
        self._rng = np.random.default_rng(seed)
        self.last_sample_cost = 0

    def __len__(self) -> int:
        return int(self._has_next[: min(self._next, self.capacity)].sum())

    @property
    def _start(self) -> int:
        return max(0, self._next - self.capacity)

    def _slot(self, absolute: int) -> int:
        return absolute % self.capacity

    def add_first(self, obs) -> None:
        slot = self._slot(self._next)
        self._frames[slot] = obs["pixels"]
        self._proprio[slot] = obs["proprio"]
        self._has_next[slot] = False
        self._age[slot] = 0
        self._last = self._next
        self._next += 1

    def add(self, action, reward, next_obs, terminated: bool, truncated: bool) -> None:
        if self._last is None:
            raise RuntimeError("call add_first() at the start of every episode")

        previous = self._slot(self._last)
        self._actions[previous] = action
        self._rewards[previous] = reward
        self._terminated[previous] = bool(terminated)
        self._truncated[previous] = bool(truncated)
        self._has_next[previous] = True

        age = self._age[previous]
        slot = self._slot(self._next)
        self._frames[slot] = next_obs["pixels"]
        self._proprio[slot] = next_obs["proprio"]
        self._has_next[slot] = False
        self._age[slot] = age + 1

        self._last = None if (terminated or truncated) else self._next
        self._next += 1

    def stack_at(self, absolute: int) -> np.ndarray:
        age = int(self._age[self._slot(absolute)])
        indices = [absolute - min(offset, age) for offset in reversed(range(self.frame_stack))]
        frames = [self._frames[self._slot(i)].transpose(2, 0, 1) for i in indices]
        return np.concatenate(frames, axis=0)

    def stacks_at(self, absolute: np.ndarray) -> np.ndarray:
        absolute = np.asarray(absolute, dtype=np.int64)
        ages = self._age[absolute % self.capacity]
        offsets = np.arange(self.frame_stack - 1, -1, -1, dtype=np.int64)
        back = np.minimum(offsets[None, :], ages[:, None])
        slots = (absolute[:, None] - back) % self.capacity
        frames = self._frames[slots]
        b, k, h, w, c = frames.shape
        return frames.transpose(0, 1, 4, 2, 3).reshape(b, k * c, h, w)

    def _is_sampleable(self, absolute: int) -> bool:
        slot = self._slot(absolute)
        if not self._has_next[slot]:
            return False
        age = int(self._age[slot])
        return absolute - min(self.frame_stack - 1, age) >= self._start

    def _sampleable_mask(self, absolute: np.ndarray) -> np.ndarray:
        slots = absolute % self.capacity
        reach = np.minimum(self.frame_stack - 1, self._age[slots])
        return self._has_next[slots] & (absolute - reach >= self._start)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        span = self._next - self._start
        if span <= 0:
            raise ValueError("buffer holds no complete transitions yet")

        out = np.empty(batch_size, dtype=np.int64)
        filled = 0
        examined = 0

        for _round in range(_MAX_REJECTION_ROUNDS):
            if filled >= batch_size:
                break
            draw = self._rng.integers(self._start, self._next, size=batch_size)
            examined += draw.size
            valid = draw[self._sampleable_mask(draw)]
            take = min(valid.size, batch_size - filled)
            out[filled : filled + take] = valid[:take]
            filled += take

        if filled < batch_size:
            available = np.fromiter(
                (i for i in range(self._start, self._next) if self._is_sampleable(i)),
                dtype=np.int64,
            )
            examined += span
            if available.size == 0:
                raise ValueError("buffer holds no complete transitions yet")
            out[filled:] = self._rng.choice(
                available, size=batch_size - filled, replace=True
            )

        self.last_sample_cost = examined
        return out

    def nstep_from(self, absolute: int) -> dict[str, float | int | bool]:
        total = 0.0
        steps = 0
        index = absolute
        terminated = False
        truncated = False

        for offset in range(self.nstep):
            slot = self._slot(index)
            if not self._has_next[slot]:
                break
            total += (self.discount**offset) * float(self._rewards[slot])
            steps += 1
            index += 1
            terminated = bool(self._terminated[slot])
            truncated = bool(self._truncated[slot])
            if terminated or truncated:
                break

        if steps == 0:
            raise ValueError(f"index {absolute} holds no complete transition")

        return {
            "reward": total,
            "steps": steps,
            "final": index,
            "discount": self.discount**steps,
            "terminated": terminated,
            "truncated": truncated,
        }

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = self.sample_indices(batch_size)
        slots = np.array([self._slot(i) for i in indices])
        windows = [self.nstep_from(i) for i in indices]
        finals = [int(window["final"]) for window in windows]
        terminated = np.array([window["terminated"] for window in windows], dtype=bool)

        return {
            "pixels": self.stacks_at(indices),
            "proprio": self._proprio[slots].copy(),
            "action": self._actions[slots].copy(),
            "reward": np.array([window["reward"] for window in windows], dtype=np.float32),
            "discount": np.array([window["discount"] for window in windows], dtype=np.float32),
            "next_pixels": self.stacks_at(np.asarray(finals)),
            "next_proprio": self._proprio[[self._slot(i) for i in finals]].copy(),
            "terminated": terminated,
            "truncated": np.array([window["truncated"] for window in windows], dtype=bool),
            "bootstrap": (~terminated).astype(np.float32),
            "steps": np.array([window["steps"] for window in windows], dtype=np.int32),
        }

    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self._frames, self._proprio, self._actions, self._rewards,
                self._terminated, self._truncated, self._has_next, self._age,
            )
        )

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "frames": self._frames,
            "proprio": self._proprio,
            "actions": self._actions,
            "rewards": self._rewards,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "has_next": self._has_next,
            "age": self._age,
        }

    def save(self, path) -> None:
        import pathlib

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "geometry": {
                "capacity": self.capacity,
                "image_size": self.image_size,
                "frame_stack": self.frame_stack,
                "nstep": self.nstep,
                "discount": self.discount,
                "proprio_dim": int(self._proprio.shape[1]),
                "action_dim": int(self._actions.shape[1]),
            },
            "next": self._next,
            "last": self._last,
            "rng": self._rng.bit_generator.state,
        }

        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        for name, array in self._arrays().items():
            np.save(temporary / f"{name}.npy", array)
        np.save(temporary / "meta.npy", np.array(meta, dtype=object))

        if path.exists():
            shutil.rmtree(path)
        temporary.replace(path)

    def load(self, path) -> None:
        import pathlib

        path = pathlib.Path(path)
        meta = np.load(path / "meta.npy", allow_pickle=True).item()

        if True:
            expected = {
                "capacity": self.capacity,
                "image_size": self.image_size,
                "frame_stack": self.frame_stack,
                "nstep": self.nstep,
                "discount": self.discount,
                "proprio_dim": int(self._proprio.shape[1]),
                "action_dim": int(self._actions.shape[1]),
            }
            found = meta["geometry"]
            if found != expected:
                differences = {
                    k: (found.get(k), v) for k, v in expected.items() if found.get(k) != v
                }
                raise ValueError(
                    f"replay snapshot does not match this run: {differences} "
                    "(found, expected). Refusing to restore it."
                )

            for name, target in self._arrays().items():
                source = np.load(path / f"{name}.npy", mmap_mode="r")
                for lo in range(0, len(target), _COPY_CHUNK):
                    target[lo : lo + _COPY_CHUNK] = source[lo : lo + _COPY_CHUNK]
                del source

        self._next = int(meta["next"])
        self._last = meta["last"]
        self._rng.bit_generator.state = meta["rng"]
