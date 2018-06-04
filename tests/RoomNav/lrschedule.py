import numpy as np

def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000):
  warmup_steps = float(warmup_steps)
  step = global_step + 1
  lr = init_lr * warmup_steps**0.5 * np.minimum(
      step * warmup_steps**-1.5, step**-0.5)

  return lr


def step_learning_rate_decay(init_lr, global_step, 
                             anneal_rate=0.98,
                             anneal_interval=10000):
  return init_lr * anneal_rate ** (global_step // anneal_interval)
