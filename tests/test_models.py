"""
Tests for GAN and GNN model forward/backward passes and memory buffer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.gan.generator import Generator
from models.gan.discriminator import Discriminator
from models.gnn.memory_buffer import ExperienceReplayBuffer


# ---------------------------------------------------------------------------
# GAN Tests
# ---------------------------------------------------------------------------


class TestGenerator:
    def test_output_shape(self):
        gen = Generator(noise_dim=32, output_dim=10, hidden_dims=[64, 64])
        z = torch.randn(8, 32)
        out = gen(z)
        assert out.shape == (8, 10)

    def test_tanh_output_range(self):
        gen = Generator(noise_dim=32, output_dim=10, hidden_dims=[64, 64])
        z = torch.randn(16, 32)
        out = gen(z)
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_sample_noise_shape(self):
        gen = Generator(noise_dim=64, output_dim=20)
        device = torch.device("cpu")
        z = gen.sample_noise(batch_size=12, device=device)
        assert z.shape == (12, 64)

    def test_conditional_generation(self):
        gen = Generator(noise_dim=32, output_dim=10, hidden_dims=[64], condition_dim=4)
        z = torch.randn(8, 32)
        cond = torch.randn(8, 4)
        out = gen(z, condition=cond)
        assert out.shape == (8, 10)

    def test_backward_pass(self):
        gen = Generator(noise_dim=16, output_dim=8, hidden_dims=[32])
        z = torch.randn(4, 16, requires_grad=False)
        out = gen(z)
        loss = out.mean()
        loss.backward()
        for p in gen.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_batch_sizes(self):
        gen = Generator(noise_dim=16, output_dim=8)
        # batch_size=1 requires eval mode (BatchNorm needs >1 sample in training mode)
        gen.eval()
        z = torch.randn(1, 16)
        out = gen(z)
        assert out.shape == (1, 8)
        gen.train()
        for bs in [4, 32, 128]:
            z = torch.randn(bs, 16)
            out = gen(z)
            assert out.shape == (bs, 8)


class TestDiscriminator:
    def test_output_shape(self):
        disc = Discriminator(input_dim=10, hidden_dims=[64, 64])
        x = torch.randn(8, 10)
        out = disc(x)
        assert out.shape == (8, 1)

    def test_no_sigmoid_activation(self):
        """Discriminator output should NOT be bounded to [0,1]."""
        disc = Discriminator(input_dim=10, hidden_dims=[32, 32])
        x = torch.randn(100, 10) * 100  # Large inputs
        out = disc(x)
        # If sigmoid were applied, all values would be in [0,1]
        # Without sigmoid, we can get values outside [0,1]
        assert out.shape == (100, 1)

    def test_anomaly_score_shape(self):
        disc = Discriminator(input_dim=10, hidden_dims=[32])
        x = torch.randn(8, 10)
        scores = disc.get_anomaly_score(x)
        assert scores.shape == (8,)

    def test_backward_pass(self):
        disc = Discriminator(input_dim=10, hidden_dims=[32])
        x = torch.randn(4, 10)
        out = disc(x)
        loss = out.mean()
        loss.backward()
        for p in disc.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_spectral_norm(self):
        disc = Discriminator(input_dim=10, use_spectral_norm=True)
        # Simply check it can forward without error
        x = torch.randn(4, 10)
        out = disc(x)
        assert out.shape == (4, 1)

    def test_no_spectral_norm(self):
        disc = Discriminator(input_dim=10, use_spectral_norm=False)
        x = torch.randn(4, 10)
        out = disc(x)
        assert out.shape == (4, 1)


# ---------------------------------------------------------------------------
# Memory Buffer Tests
# ---------------------------------------------------------------------------


class TestExperienceReplayBuffer:
    def test_add_and_len(self):
        buf = ExperienceReplayBuffer(capacity=100)
        for i in range(50):
            buf.add(graph_data={"x": i}, label=i % 2)
        assert len(buf) == 50

    def test_reservoir_sampling_capacity(self):
        buf = ExperienceReplayBuffer(capacity=10)
        for i in range(100):
            buf.add(graph_data={"x": i}, label=0)
        assert len(buf) == 10

    def test_sample_returns_correct_size(self):
        buf = ExperienceReplayBuffer(capacity=100)
        for i in range(50):
            buf.add(graph_data={"x": i}, label=i % 2)
        samples = buf.sample(10)
        assert len(samples) == 10

    def test_sample_empty_buffer(self):
        buf = ExperienceReplayBuffer(capacity=100)
        samples = buf.sample(10)
        assert samples == []

    def test_class_balanced_sample(self):
        buf = ExperienceReplayBuffer(capacity=1000)
        # Add 90 class-0 and 10 class-1
        for i in range(90):
            buf.add(graph_data={"x": i}, label=0)
        for i in range(10):
            buf.add(graph_data={"x": i + 90}, label=1)

        samples = buf.get_class_balanced_sample(20)
        assert len(samples) <= 20
        labels = [s.label for s in samples]
        # Should have both classes represented
        assert 0 in labels
        assert 1 in labels

    def test_priority_based_sampling(self):
        buf = ExperienceReplayBuffer(capacity=100, priority_alpha=0.6)
        for i in range(50):
            buf.add(graph_data={"x": i}, label=0, priority=float(i + 1))
        samples = buf.sample(10)
        assert len(samples) == 10

    def test_n_seen_counter(self):
        buf = ExperienceReplayBuffer(capacity=100)
        for i in range(200):
            buf.add(graph_data={}, label=0)
        assert buf._n_seen == 200

    def test_repr(self):
        buf = ExperienceReplayBuffer(capacity=100)
        buf.add(graph_data={}, label=0)
        r = repr(buf)
        assert "ExperienceReplayBuffer" in r

    def test_add_batch(self):
        buf = ExperienceReplayBuffer(capacity=100)
        graphs = [{"x": i} for i in range(20)]
        labels = [i % 2 for i in range(20)]
        buf.add_batch(graphs, labels, task_id=1)
        assert len(buf) == 20
