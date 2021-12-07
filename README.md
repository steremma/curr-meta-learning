# Curriculum Meta-Learning for Few-shot Classification

Link to the paper: https://arxiv.org/abs/2112.02913

 We propose an adaptation of the curriculum training framework, applicable to state-of-the-art meta learning techniques for few-shot classification. Curriculum-based training popularly attempts to mimic human learning by progressively increasing the training complexity to enable incremental concept learning. As the meta-learner's goal is learning how to learn from as few samples as possible, the exact number of those samples (i.e. the size of the support set) arises as a natural proxy of a given task's difficulty. We define a simple yet novel curriculum schedule that begins with a larger support size and progressively reduces it throughout training to eventually match the desired shot-size of the test setup. This proposed method boosts the learning efficiency as well as the generalization capability. Our experiments with the MAML algorithm on two few-shot image classification tasks show significant gains with the curriculum training framework. Ablation studies corroborate the independence of our proposed method from the model architecture as well as the meta-learning hyperparameters.

# How to reproduce
Our code is based on the [learn2learn](https://github.com/learnables/learn2learn) library. Specifically we start from their MAML implementation and extend with the ideas presented in our paper. Each of the results presented in the paper (incl. Ablation studies) can be reproduced by invoking the main script with appropriate arguments. 

#### Requirements
Install dependencies:
```
pip install torch
pip install learn2learn
```

## Examples

#### 5 way - 5 shot MiniImagenet using a Convolutional neural network.
```
# Vanilla, achieves ~ 58% accuracy
python3 curriculum_meta_learning.py --dataset mini-imagenet --multiplier 1 --shot 5 --ways 5

# Ours, achieves ~ 66% accuracy
python3 curriculum_meta_learning.py --dataset mini-imagenet --multiplier 5 --shot 5 --ways 5
```

#### 5 way - 1 shot OmniGlot using a Fully Connected neural network.
```
# Vanilla, achieves ~ 90% accuracy
python3 curriculum_meta_learning.py --dataset omniglot --multiplier 1 --shot 1 --ways 5 --fc

# Ours, achieves ~ 94% accuracy
python3 curriculum_meta_learning.py --dataset omniglot --multiplier 5 --shot 1 --ways 5 --fc
```

#### Ablation: disable LR annealing or query size adaptation durinng training.
```
python3 curriculum_meta_learning.py --multiplier 3 --freeze_lr

python3 curriculum_meta_learning.py --multiplier 3 --freeze_l
```

#### Ablation: Use a statically larger support size instead of curriculum.
```
python3 curriculum_meta_learning.py --dataset mini-imagenet --multiplier 5 --shot 5 --ways 5 --freeze_multiplier
```


 ## Authors
* Stergiadis Emmanouil (@steremma), emmanouil.stergiadis@booking.com
* Priyanka Agrawal (@pagrawal-ml), priyanka.agrawal@booking.com
* Oliver Squire (@ojsquire), oliver.squire@booking.com
