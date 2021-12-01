#!/usr/bin/env python3

"""
Fork of `learn2learn/examples/vision/maml_miniimagenet.py`.
"""

import random
import argparse
import math
import numpy as np

import torch
from torch import nn, optim

import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, freeze_l):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation / evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(round(data.size(0) / 2)) * 2] = True

    # Freeze query size.
    if freeze_l:
        stride = round(data.size(0) / (ways * shots))
        evaluation_indices = np.zeros(data.size(0), dtype=bool)
        evaluation_indices[np.arange(ways * shots) * stride + 1] = True
    else:
        evaluation_indices = ~adaptation_indices

    evaluation_indices = torch.from_numpy(evaluation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def main(
        dataset,
        ways=5,
        shots=5,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=100,
        eval_every=300,
        multiplier=1,
        cnn=False,
        resnet=False,
        freeze_l=False,
        freeze_lr=False,
        freeze_multiplier=False
):
    print("Running on {}, {} way - {} shot".format(dataset, ways, shots), flush=True)
    print("Multiplier = {}, Freeze L = {}, Freeze LR = {}, Freeze Multiplier = {}".format(multiplier, freeze_l, freeze_lr, freeze_multiplier), flush=True)
    if dataset == "omniglot":
        print("Using {} architecture".format("CONV" if cnn else "FC"), flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        print("Using GPU", flush=True)
        device = torch.device('cuda')
    else:
        print("Using CPU", flush=True)

    # Create model
    if not cnn:
        model = l2l.vision.models.MiniImagenetCNN(ways) if dataset == "mini-imagenet" else l2l.vision.models.OmniglotFC(28 ** 2, ways)
    else:
        model = l2l.vision.models.MiniImagenetCNN(ways) if dataset == "mini-imagenet" else l2l.vision.models.OmniglotCNN(ways)
    if resnet:
        model = l2l.vision.models.ResNet12(output_size=ways)
    model.to(device)

    # Adapt inner loop learning rate
    if not freeze_lr:
        initial_fast_lr = fast_lr
        fast_lr *= math.sqrt(multiplier)

    print("Starting with inner loop LR = {}".format(fast_lr), flush=True)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    # Create Tasksets using the benchmark interface
    # In contrast to master, we recreate the dataset with different batch size in each training iteration.
    # The reason is we wish to vary the support (and query) size.
    # Note that redrawing here should not introduce a leak because the method does NOT randomise itself
    # to create the splits. Instead those splits are hardcoded upstream to facilitate fair comparisons.
    # So at every iteration the validation and test set batches are sampled from the same splits.
    training_samples = 2 * shots * multiplier

    # The plan is to complete the curriculum over half the iterations. The other half of the training will
    # proceed vanilla (K=shot).
    switch_stage_every = num_iterations if multiplier < 2 else round((num_iterations / 2) / (multiplier - 1))
    print("We start with {} training samples (half of which are support)".format(training_samples), flush=True)
    print("We shall switch stage every {} iterations".format(switch_stage_every), flush=True)
    tasksets = l2l.vision.benchmarks.get_tasksets(dataset,
                                                  train_samples=training_samples,
                                                  train_ways=ways,
                                                  test_samples=2 * shots,
                                                  test_ways=ways,
                                                  root='~/data')

    # Support and query sets.
    best_meta_valid_error = float("inf")
    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        if iteration % eval_every == 0:
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device,
                                                               freeze_l)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            if iteration % eval_every == 0:
                learner = maml.clone()
                batch = tasksets.validation.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   shots,
                                                                   ways,
                                                                   device,
                                                                   True)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        if iteration % eval_every == 0:
            meta_valid_error = meta_valid_error / meta_batch_size
            if meta_valid_error < best_meta_valid_error:
                best_meta_valid_error = meta_valid_error
            else:
                print("Early stopping here?", flush=True)

            print('\n', flush=True)
            print('Iteration', iteration, flush=True)
            print('Meta Train Error', meta_train_error / meta_batch_size, flush=True)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size, flush=True)
            print('Meta Valid Error', meta_valid_error, flush=True)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size, flush=True)

        if iteration % switch_stage_every == 0 and iteration > 0 and not freeze_multiplier:
            training_samples -= 2 * shots
            training_samples = max(training_samples, shots * 2)
            # Adapt inner loop learning rate
            if not freeze_lr:
                fast_lr = math.sqrt(fast_lr**2 - initial_fast_lr**2)
                fast_lr = max(fast_lr, initial_fast_lr)
                print("Learning rate now set to {}".format(fast_lr), flush=True)
                maml.lr = fast_lr
            print("Creating new dataset with {} training samples".format(training_samples), flush=True)
            print("Learning rate is now {:.2f}".format(fast_lr), flush=True)
            tasksets = l2l.vision.benchmarks.get_tasksets(dataset,
                                                          train_samples=training_samples,
                                                          train_ways=ways,
                                                          test_samples=2 * shots,
                                                          test_ways=ways,
                                                          root='~/data',
                                                          )

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device,
                                                           True)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()

    print('Meta Test Error', meta_test_error / meta_batch_size, flush=True)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size, flush=True)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiplier', default=1,  type=int, help="Shot multiplier for maximum (=initial) support size.")
    parser.add_argument('--shots', default=5,  type=int, help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--ways', default=5,  type=int, help="Number of candidate labels at meta-test time")
    parser.add_argument('--freeze_l', action='store_true', help="Should L be frozen to original value instead of matching support size?")
    parser.add_argument('--freeze_lr', action='store_true', help="Static inner loop lr, or normalised by root of batch size?")
    parser.add_argument('--freeze_multiplier', action='store_true', help="Freeze the support size to multiplier * shot for ablation study.")
    parser.add_argument('--dataset', default="mini-imagenet", choices=["mini-imagenet", "omniglot"], help="Dataset to use.")
    parser.add_argument('--fc', action='store_true', help="Use fully connected rather than convolutional back-bone. Only relevant for omniglot.")
    parser.add_argument('--resnet', action='store_true', help="Use resnet.")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(dataset=args.dataset, multiplier=args.multiplier,
         freeze_l=args.freeze_l, freeze_lr=args.freeze_lr,
         freeze_multiplier=args.freeze_multiplier,
         shots=args.shots, ways=args.ways,
         cnn=not args.fc, resnet=args.resnet)
