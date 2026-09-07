"""Differentiable Ebbinghaus forgetting-curve memory.

Each interaction is stored as a trace with a stability value S. Its retention
at age t follows the classic exponential curve R(t) = exp(-t / S). The
retrieval logits combine retention with cue similarity so the memory can be
trained end to end inside GKT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EbbinghausMemory(nn.Module):
    """Bounded trace memory using exponential forgetting."""

    def __init__(self, input_dim, memory_dim, max_traces=64, stability=1.0,
                 temperature=0.2):
        super().__init__()
        if max_traces < 1:
            raise ValueError("max_traces must be positive")
        if stability <= 0:
            raise ValueError("stability must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.max_traces = max_traces
        self.stability = stability
        self.temperature = temperature
        self.trace_encoder = nn.Linear(input_dim, memory_dim)
        self.query_encoder = nn.Linear(input_dim, memory_dim)
        self.stability_encoder = nn.Linear(input_dim, 1)

    def _read(self, traces, trace_times, trace_stability, valid, query, now):
        query_key = F.normalize(self.query_encoder(query), dim=-1)
        trace_key = F.normalize(traces, dim=-1)
        similarity = torch.einsum("bd,bnd->bn", query_key, trace_key)
        age = (now.unsqueeze(1) - trace_times).clamp_min(0)
        retention = torch.exp(-age / trace_stability.clamp_min(1e-6))
        logits = similarity / self.temperature + torch.log(retention.clamp_min(1e-8))
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        retrieved = torch.einsum("bn,bnd->bd", weights, traces)
        strength = (weights * retention).sum(dim=-1)
        return retrieved, strength, weights, retention

    def initial_state(self, batch_size, device, dtype):
        traces = torch.zeros(batch_size, self.max_traces, self.memory_dim, device=device, dtype=dtype)
        trace_times = torch.zeros(batch_size, self.max_traces, device=device, dtype=dtype)
        trace_stability = torch.full(
            (batch_size, self.max_traces), self.stability, device=device, dtype=dtype,
        )
        valid = torch.zeros(batch_size, self.max_traces, device=device, dtype=torch.bool)
        now = torch.zeros(batch_size, device=device, dtype=dtype)
        return traces, trace_times, trace_stability, valid, now

    def step(self, state, event, mask, step, elapsed=None):
        """Write one event and retrieve it using exponential retention."""
        traces, trace_times, trace_stability, valid, now = state
        if elapsed is None:
            elapsed = torch.ones_like(now)
        now = now + elapsed.clamp_min(0)
        encoded = torch.tanh(self.trace_encoder(event))
        encoded_stability = F.softplus(self.stability_encoder(event).squeeze(-1)) + self.stability
        slot = step % self.max_traces

        traces = traces.clone()
        traces[:, slot:slot + 1] = torch.where(
            mask[:, None, None], encoded[:, None, :], traces[:, slot:slot + 1],
        )
        trace_times = trace_times.clone()
        trace_times[:, slot] = torch.where(mask, now, trace_times[:, slot])
        trace_stability = trace_stability.clone()
        trace_stability[:, slot] = torch.where(
            mask, encoded_stability, trace_stability[:, slot],
        )
        valid = valid.clone()
        valid[:, slot] = valid[:, slot] | mask
        retrieved, strength, weights, retention = self._read(
            traces, trace_times, trace_stability, valid, event, now,
        )
        return (traces, trace_times, trace_stability, valid, now), retrieved, strength, weights

    def forward(self, events, mask=None, elapsed=None):
        """Read memory after each event in ``[batch, sequence, input_dim]``."""
        if events.ndim != 3 or events.size(-1) != self.input_dim:
            raise ValueError("events must have shape [batch, sequence, input_dim]")
        batch_size, sequence_length, _ = events.shape
        device = events.device
        if mask is None:
            mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
        if elapsed is None:
            elapsed = torch.ones(batch_size, sequence_length, device=device, dtype=events.dtype)
        if mask.shape != (batch_size, sequence_length):
            raise ValueError("mask must have shape [batch, sequence]")
        if elapsed.shape != (batch_size, sequence_length):
            raise ValueError("elapsed must have shape [batch, sequence]")

        state = self.initial_state(batch_size, device, events.dtype)
        retrieved, strengths, weights = [], [], []
        for step in range(sequence_length):
            state, current, strength, weight = self.step(
                state, events[:, step], mask[:, step], step, elapsed[:, step],
            )
            retrieved.append(current)
            strengths.append(strength)
            weights.append(weight)
        return torch.stack(retrieved, dim=1), torch.stack(strengths, dim=1), torch.stack(weights, dim=1)