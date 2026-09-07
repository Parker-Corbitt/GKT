"""Differentiable multiple-trace memory inspired by ATHENA.

This module implements the computational core that is useful for knowledge
tracing: every interaction becomes a trace, traces are retrieved by cue
similarity, and retrieval is modulated by a power-law recency term. It is an
engineering approximation of the ATHENA paper rather than a claim of exact
parameter recovery from the paper's psychological simulations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AthenaMemory(nn.Module):
    """Batch-first multiple-trace memory with differentiable retrieval."""

    def __init__(self, input_dim, memory_dim, max_traces=64, decay=0.5,
                 temperature=0.2):
        super().__init__()
        if max_traces < 1:
            raise ValueError("max_traces must be positive")
        if decay <= 0:
            raise ValueError("decay must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.max_traces = max_traces
        self.decay = decay
        self.temperature = temperature
        self.trace_encoder = nn.Linear(input_dim, memory_dim)
        self.query_encoder = nn.Linear(input_dim, memory_dim)

    def _read(self, traces, trace_times, valid, query, now):
        query_key = F.normalize(self.query_encoder(query), dim=-1)
        trace_key = F.normalize(traces, dim=-1)
        similarity = torch.einsum("bd,bnd->bn", query_key, trace_key)
        age = (now.unsqueeze(1) - trace_times).clamp_min(0).add(1.0)
        log_decay = -self.decay * torch.log(age)
        logits = similarity / self.temperature + log_decay
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid.to(weights.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights / normalizer
        retrieved = torch.einsum("bn,bnd->bd", weights, traces)
        strength = (weights * similarity).sum(dim=-1)
        return retrieved, strength, weights

    def initial_state(self, batch_size, device, dtype):
        traces = torch.zeros(batch_size, self.max_traces, self.memory_dim, device=device, dtype=dtype)
        trace_times = torch.zeros(batch_size, self.max_traces, device=device, dtype=dtype)
        valid = torch.zeros(batch_size, self.max_traces, device=device, dtype=torch.bool)
        now = torch.zeros(batch_size, device=device, dtype=dtype)
        return traces, trace_times, valid, now

    def step(self, state, event, mask, step, elapsed=None):
        """Write one event and retrieve its cue-conditioned memory context."""
        traces, trace_times, valid, now = state
        if elapsed is None:
            elapsed = torch.ones_like(now)
        now = now + elapsed.clamp_min(0)
        encoded = torch.tanh(self.trace_encoder(event))
        slot = step % self.max_traces
        traces = traces.clone()
        traces[:, slot:slot + 1] = torch.where(
            mask[:, None, None],
            encoded[:, None, :],
            traces[:, slot:slot + 1],
        )
        trace_times = trace_times.clone()
        trace_times[:, slot] = torch.where(mask, now, trace_times[:, slot])
        valid = valid.clone()
        valid[:, slot] = valid[:, slot] | mask
        retrieved, strength, weights = self._read(
            traces, trace_times, valid, event, now,
        )
        return (traces, trace_times, valid, now), retrieved, strength, weights

    def forward(self, events, mask=None, elapsed=None):
        """Write each event and return the memory retrieved after each write.

        Args:
            events: Tensor shaped ``[batch, sequence, input_dim]``.
            mask: Optional boolean tensor shaped ``[batch, sequence]``.
            elapsed: Optional non-negative time gaps shaped ``[batch, sequence]``.
                Without timestamps, sequence position is used as time.

        Returns:
            retrieved: Memory context after each event.
            strengths: Cue-memory similarity for each event.
            weights: Trace retrieval weights for each event.
        """
        if events.ndim != 3 or events.size(-1) != self.input_dim:
            raise ValueError("events must have shape [batch, sequence, input_dim]")
        batch_size, sequence_length, _ = events.shape
        device = events.device
        if mask is None:
            mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
        if elapsed is None:
            elapsed = torch.ones(batch_size, sequence_length, device=device)
        if mask.shape != (batch_size, sequence_length):
            raise ValueError("mask must have shape [batch, sequence]")
        if elapsed.shape != (batch_size, sequence_length):
            raise ValueError("elapsed must have shape [batch, sequence]")

        state = self.initial_state(batch_size, device, events.dtype)
        retrieved, strengths, weights = [], [], []

        for step in range(sequence_length):
            state, current, strength, weight = self.step(
                state,
                events[:, step],
                mask[:, step],
                step,
                elapsed[:, step],
            )
            retrieved.append(current)
            strengths.append(strength)
            weights.append(weight)

        return torch.stack(retrieved, dim=1), torch.stack(strengths, dim=1), torch.stack(weights, dim=1)


class AthenaKT(nn.Module):
    """A compact ATHENA-style next-response model for the project's data."""

    def __init__(self, concept_num, memory_dim=32, max_traces=64,
                 decay=0.5, dropout=0.0):
        super().__init__()
        self.concept_num = concept_num
        self.memory_dim = memory_dim
        self.concept_embedding = nn.Embedding(concept_num + 1, memory_dim)
        self.response_embedding = nn.Embedding(3, memory_dim)
        self.event_encoder = nn.Sequential(
            nn.Linear(2 * memory_dim, memory_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.memory = AthenaMemory(
            input_dim=memory_dim,
            memory_dim=memory_dim,
            max_traces=max_traces,
            decay=decay,
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * memory_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(memory_dim, 1),
        )

    def forward(self, features, questions, elapsed=None):
        """Return next-question probabilities shaped ``[batch, sequence - 1]``."""
        valid = questions.ne(-1)
        safe_questions = questions.masked_fill(~valid, self.concept_num)
        safe_features = features.masked_fill(~valid, 2).clamp_min(0).clamp_max(2)
        concepts = self.concept_embedding(safe_questions)
        responses = self.response_embedding(safe_features)
        events = self.event_encoder(torch.cat((concepts, responses), dim=-1))
        memory_context, _, _ = self.memory(events, valid, elapsed=elapsed)
        state = torch.cat((events, memory_context), dim=-1)
        all_predictions = torch.sigmoid(self.predictor(state)).squeeze(-1)
        next_questions = questions[:, 1:].clamp_min(0)
        next_valid = questions[:, 1:].ne(-1)
        next_concepts = self.concept_embedding(next_questions)
        next_context = memory_context[:, :-1]
        next_state = torch.cat((next_concepts, next_context), dim=-1)
        predictions = torch.sigmoid(self.predictor(next_state)).squeeze(-1)
        predictions = predictions.masked_fill(~next_valid, 0.0)
        return predictions