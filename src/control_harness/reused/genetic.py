"""Extracted numerical code. See reuse-manifest.json; original registries are excluded."""
from __future__ import annotations
import math
import random
from typing import Sequence
import torch
from torch import nn

class GeneticOperators:

    @staticmethod
    def _fitness(trial: dict[str, object]) -> float:
        if not trial.get('hard_constraints_passed') or not isinstance(trial.get('score'), (int, float)):
            return math.inf
        return float(trial['score'])

    @staticmethod
    def _initial_population(space: dict[str, dict[str, object]], size: int, rng: random.Random, current: dict[str, object]) -> list[dict[str, float]]:
        population = [{name: float(current[name]) for name in space}]
        while len(population) < size:
            item = {}
            for (name, spec) in space.items():
                (low, high) = (float(spec['lower']), float(spec['upper']))
                value = rng.uniform(low, high)
                item[name] = float(int(round(value))) if spec.get('integer') is True else value
            population.append(item)
        return population

    @staticmethod
    def _tournament(ranked: list[dict[str, object]], rng: random.Random) -> dict[str, object]:
        contenders = rng.sample(ranked, min(3, len(ranked)))
        return min(contenders, key=GeneticOperators._fitness)

    @staticmethod
    def _crossover(a: dict[str, object], b: dict[str, object], space: dict[str, dict[str, object]], rng: random.Random) -> tuple[dict[str, float], dict[str, object]]:
        alpha = rng.random()
        child = {}
        for (name, spec) in space.items():
            value = alpha * float(a[name]) + (1 - alpha) * float(b[name])
            child[name] = float(int(round(value))) if spec.get('integer') is True else value
        return (child, {'alpha': alpha})

    @staticmethod
    def _mutate(candidate: dict[str, float], space: dict[str, dict[str, object]], rng: random.Random) -> tuple[dict[str, float], list[dict[str, object]]]:
        result = dict(candidate)
        events = []
        names = list(space)
        forced = rng.choice(names)
        for (name, spec) in space.items():
            if name != forced and rng.random() >= 0.3:
                continue
            (low, high) = (float(spec['lower']), float(spec['upper']))
            sigma = max((high - low) * 0.12, 1e-12)
            before = result[name]
            value = min(high, max(low, before + rng.gauss(0, sigma)))
            if spec.get('integer') is True:
                value = float(int(round(value)))
            result[name] = value
            events.append({'parameter': name, 'before': before, 'after': value, 'sigma': sigma})
        return (result, events)
