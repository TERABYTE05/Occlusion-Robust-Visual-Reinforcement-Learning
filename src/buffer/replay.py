"""Lazy uint8 replay buffer.

The paper's buffer is 10^6 float32 three-frame stacks, which is 254 GB. Ours is
10^5 transitions of *single* uint8 frames, rebuilt into stacks at sample time:
84*84*3 bytes x 10^5 = 2.12 GB, which fits in 16 GB alongside a second run.
Storing stacks instead would put each frame in the buffer three times and take
the memory back.

Two details carry real weight.

Stacks must not reach across an episode boundary. Slot ``i`` records how many
steps into its episode it is, so a stack at the start of an episode repeats the
first frame instead of pulling in the tail of the previous one. Getting this
wrong is invisible -- the shapes are identical either way -- and it poisons a
small fraction of every batch.

``terminated`` and ``truncated`` are stored separately and never collapsed into
a single ``done``. FetchPush runs a fixed 50 steps and never sets ``terminated``,
so every episode ends by truncation; a buffer that only kept ``done`` would have
thrown away the distinction the bootstrap mask depends on (Landmine 6).

The n-step return is accumulated here rather than in the agent, because it is the
only place that knows where episodes end. Three cases, and the difference between
the last two is Landmine 6:

- the window runs its full ``nstep`` steps: reward is the discounted sum,
  bootstrap from the observation ``nstep`` later, discounted by ``gamma**nstep``;
- the episode **truncates** inside the window: the sum stops there, because there
  are no further rewards, but the value target still **bootstraps normally** from
  the final observation. Truncation is an artificial time limit, not an absorbing
  state, and zeroing the target there would misprice every state near the end of
  an episode -- which on FetchPush is every episode, all of them;
- the episode **terminates** inside the window: the sum stops and the bootstrap is
  **zeroed**, because there is genuinely no future value.

So ``bootstrap`` is ``1 - terminated``, never ``1 - (terminated or truncated)``,
and ``discount`` is ``gamma ** steps_actually_taken`` rather than a constant.
"""

from __future__ import annotations

import numpy as np

#: Rejection rounds before falling back to an exhaustive scan. At ~98%
#: acceptance a single round almost always fills the batch; this only matters
#: for a buffer too small or too sparse for rejection to converge.
_MAX_REJECTION_ROUNDS = 8


class ReplayBuffer:
    """Ring buffer over observations, not over transitions.

    Each slot holds one observation plus the action taken from it, the reward
    that action earned, and how the episode ended if it ended there. A
    transition is a slot together with its successor, so consecutive
    observations are stored once rather than twice.
    """

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
        #: whether this slot has an action and a successor observation recorded
        self._has_next = np.zeros(capacity, dtype=bool)
        #: steps into the episode, so stacks can be clamped at its start
        self._age = np.zeros(capacity, dtype=np.int32)

        self._next = 0  # absolute index of the next write
        self._last: int | None = None  # absolute index of the open observation
        self._rng = np.random.default_rng(seed)
        #: Indices inspected by the most recent ``sample_indices`` call.
        self.last_sample_cost = 0

    # -- bookkeeping ------------------------------------------------------
    def __len__(self) -> int:
        """Number of slots holding a complete transition."""
        return int(self._has_next[: min(self._next, self.capacity)].sum())

    @property
    def _start(self) -> int:
        """Lowest absolute index not yet overwritten."""
        return max(0, self._next - self.capacity)

    def _slot(self, absolute: int) -> int:
        return absolute % self.capacity

    # -- writing ----------------------------------------------------------
    def add_first(self, obs) -> None:
        """Record the observation returned by ``env.reset()``."""
        slot = self._slot(self._next)
        self._frames[slot] = obs["pixels"]
        self._proprio[slot] = obs["proprio"]
        self._has_next[slot] = False
        self._age[slot] = 0
        self._last = self._next
        self._next += 1

    def add(self, action, reward, next_obs, terminated: bool, truncated: bool) -> None:
        """Record a step. Call ``add_first`` again after an episode ends."""
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

    # -- reading ----------------------------------------------------------
    def stack_at(self, absolute: int) -> np.ndarray:
        """The ``frame_stack`` frames ending at ``absolute``, as (3k, H, W).

        Clamped at the start of the episode: an observation ``t`` steps in
        repeats its first frame rather than borrowing the previous episode's.
        """
        age = int(self._age[self._slot(absolute)])
        indices = [absolute - min(offset, age) for offset in reversed(range(self.frame_stack))]
        frames = [self._frames[self._slot(i)].transpose(2, 0, 1) for i in indices]
        return np.concatenate(frames, axis=0)

    def stacks_at(self, absolute: np.ndarray) -> np.ndarray:
        """``stack_at`` for a whole batch, as ``(B, 3k, H, W)``.

        One fancy-index gather instead of a Python loop with a transpose and a
        concatenate per sample. At 84x84 the loop form cost ~19 ms for a batch of
        256 (twice, for obs and next_obs), which is a quarter of the training
        step -- invisible at the 1x1 frame size the state anchor uses, which is
        how it survived the first round of profiling.
        """
        absolute = np.asarray(absolute, dtype=np.int64)
        ages = self._age[absolute % self.capacity]
        offsets = np.arange(self.frame_stack - 1, -1, -1, dtype=np.int64)
        # clamp at the episode start exactly as stack_at does
        back = np.minimum(offsets[None, :], ages[:, None])
        slots = (absolute[:, None] - back) % self.capacity
        frames = self._frames[slots]                      # (B, k, H, W, 3)
        b, k, h, w, c = frames.shape
        return frames.transpose(0, 1, 4, 2, 3).reshape(b, k * c, h, w)

    def _is_sampleable(self, absolute: int) -> bool:
        slot = self._slot(absolute)
        if not self._has_next[slot]:
            return False
        # the oldest frame the stack needs must still be present
        age = int(self._age[slot])
        return absolute - min(self.frame_stack - 1, age) >= self._start

    def _sampleable_mask(self, absolute: np.ndarray) -> np.ndarray:
        """``_is_sampleable`` over a whole array at once, in numpy."""
        slots = absolute % self.capacity
        reach = np.minimum(self.frame_stack - 1, self._age[slots])
        return self._has_next[slots] & (absolute - reach >= self._start)

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Absolute indices of complete, fully-reconstructable transitions.

        Drawn by **rejection**: propose ``batch_size`` indices uniformly over
        the live span, keep the ones that pass, repeat until the batch is full.
        The distribution is identical to picking uniformly from the set of valid
        indices, because a uniform proposal filtered by a predicate is uniform
        over what survives it.

        The obvious implementation -- build the list of valid indices, then
        choose from it -- is O(buffer) on *every* training step, and the buffer
        holds 10^5 transitions. Measured on the state anchor, that put a ceiling
        of ~31 FPS on the whole run once the buffer filled, decaying steadily
        from ~240 FPS as it filled up, and it is invisible in
        ``benchmark_fps.py`` because that script uses a resident batch and
        explicitly does not measure sampling. Throughput decides the run matrix
        (G1, G4), so this is a scientific cost, not a micro-optimisation.

        Rejection is cheap here because almost everything is valid: only the
        final observation of each episode lacks a successor, and only a couple
        of indices at the tail of the ring have lost the history their stack
        needs. On 50-step episodes that is an acceptance rate around 98%.
        """
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
            # Pathological: a buffer so short or so sparse that rejection is not
            # converging. Fall back to the exhaustive scan, which is correct and
            # only ever runs when the buffer is tiny anyway.
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

        #: Indices inspected by the last call. A diagnostic, like ``nbytes``:
        #: it must stay proportional to the batch, never to the buffer.
        self.last_sample_cost = examined
        return out

    def nstep_from(self, absolute: int) -> dict[str, float | int | bool]:
        """Walk the n-step window starting at ``absolute``.

        Returns the discounted reward sum, how many steps it actually covered,
        the absolute index of the observation to bootstrap from, and how the
        window ended. The walk stops early at a terminal, at a truncation, or
        at the write frontier; in all three cases the shortened return plus a
        ``gamma ** steps`` bootstrap is still the correct target, which is why
        ``discount`` is returned rather than assumed.
        """
        total = 0.0
        steps = 0
        index = absolute
        terminated = False
        truncated = False

        for offset in range(self.nstep):
            slot = self._slot(index)
            if not self._has_next[slot]:
                break  # write frontier: the successor has not been recorded yet
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
        """A batch of n-step transitions.

        ``reward`` is the discounted sum over the window and ``discount`` is
        ``gamma ** steps``, so the value target is

            reward + discount * bootstrap * Q(next)

        ``bootstrap`` is ``1 - terminated``. ``truncated`` is returned too, but
        only for diagnostics -- it must never enter the mask (Landmine 6).
        """
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

    # -- diagnostics ------------------------------------------------------
    def nbytes(self) -> int:
        """Resident size of the stored arrays, for the week-4 memory check."""
        return sum(
            array.nbytes
            for array in (
                self._frames, self._proprio, self._actions, self._rewards,
                self._terminated, self._truncated, self._has_next, self._age,
            )
        )

    # -- persistence ------------------------------------------------------
    def save(self, path) -> None:
        """Write the buffer so a resumed run keeps its replay history.

        Without this, ``--resume`` rebuilds an empty buffer and the run trains
        on near-on-policy data until it refills. A 500K run restarted at 150K
        would throw away 100K transitions and would no longer be the same
        experiment as an uninterrupted one -- silently, since nothing in the
        curves says a restart happened. Runs are hours long on a shared
        machine and the roadmap expects restarts, so this has to be real.
        """
        import pathlib

        import torch

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "geometry": {
                "capacity": self.capacity,
                "image_size": self.image_size,
                "frame_stack": self.frame_stack,
                "nstep": self.nstep,
                "discount": self.discount,
                "proprio_dim": int(self._proprio.shape[1]),
                "action_dim": int(self._actions.shape[1]),
            },
            "frames": self._frames,
            "proprio": self._proprio,
            "actions": self._actions,
            "rewards": self._rewards,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "has_next": self._has_next,
            "age": self._age,
            "next": self._next,
            "last": self._last,
            "rng": self._rng.bit_generator.state,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path) -> None:
        """Restore a buffer written by :meth:`save`, in place.

        The geometry is checked rather than trusted. A buffer restored into a
        run with a different frame stack or capacity would reconstruct stacks
        that never existed, which is exactly the kind of silent corruption the
        rest of this class is built to prevent.
        """
        import pathlib

        import torch

        payload = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)

        expected = {
            "capacity": self.capacity,
            "image_size": self.image_size,
            "frame_stack": self.frame_stack,
            "nstep": self.nstep,
            "discount": self.discount,
            "proprio_dim": int(self._proprio.shape[1]),
            "action_dim": int(self._actions.shape[1]),
        }
        found = payload["geometry"]
        if found != expected:
            differences = {k: (found.get(k), v) for k, v in expected.items() if found.get(k) != v}
            raise ValueError(
                f"replay snapshot does not match this run: {differences} "
                "(found, expected). Refusing to restore it."
            )

        self._frames[...] = payload["frames"]
        self._proprio[...] = payload["proprio"]
        self._actions[...] = payload["actions"]
        self._rewards[...] = payload["rewards"]
        self._terminated[...] = payload["terminated"]
        self._truncated[...] = payload["truncated"]
        self._has_next[...] = payload["has_next"]
        self._age[...] = payload["age"]
        self._next = int(payload["next"])
        self._last = payload["last"]
        self._rng.bit_generator.state = payload["rng"]
