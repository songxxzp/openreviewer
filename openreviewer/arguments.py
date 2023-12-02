import argparse
import os
import deepspeed
import datetime


def add_model_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('model', 'model arguments')
    group.add_argument("--model-path", type=str, default=None)
    group.add_argument("--model-type", type=str, default=None)

    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data arguments')
    group.add_argument("--data-path", type=str, default=None)
    group.add_argument("--data-name", type=str, default=None)
    group.add_argument("--dataset-type", type=str, default="InstructionTuningDataset")
    group.add_argument("--max-length", type=int, default=1024)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--num-workers", type=int, default=4)

    return parser


def add_train_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('training', 'training arguments')
    group.add_argument("--total-iters", type=int, default=100)
    group.add_argument("--warmup-steps", type=int, default=100)
    group.add_argument("--batch-size", type=int, default=2)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--save-path", type=str, default=None)

    return parser


def add_optimizer_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('distribution', 'distribution arguments')
    group.add_argument("--lr", type=float, default=2e-5)
    group.add_argument("--min-lr", type=float, default=1e-6)
    group.add_argument("--weight-decay", type=float, default=1e-1)
    group.add_argument("--clip-grad", type=float, default=1.0)
    group.add_argument('--seed', type=int, default=42, help='Good luck')

    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_data_args(parser)
    parser = add_train_args(parser)
    parser = add_optimizer_args(parser)
    parser = deepspeed.add_config_arguments(parser)

    args, unknown = parser.parse_known_args()

    print(f"Unkown args: {unknown}")
    print(f"args: {args}")

    return args
