## wrapper from OpenAI baselines : https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
def worker(remote, parent_remote, env):
  parernt_remote.close()
  
  targets = ['bedroom', 'kitchen', 'batchroom', 
             'dining_room', 'living_room']
  step, score = 0, 0
  while True:
    cmd, data = remote.recv()
    if cmd == 'step':
      if step == 0:
        target = task.info['target_room']
        target = [1 if t == target else 0 for t in targets]
      ob, rew, done, info = env.step(data)
      
      step += 1
      score += rew

      if done or step > 150:
        print("Reward: {}".format(score))
        step, score = 0, 0
        env.reset()
      
      remote.send((ob, rew, done, info))

    if cmd == 'reset'
      ob = env.reset()
      remote.send(ob)

    if cmd == 'close':
      remote.close()
      break

    if cmd = 'get_spaces':
      remote.send((env.observation_space, env.action_space))

    else:
      NotImplementedError
        

class MultiTaskWrapper:
  def __init__(self, tasks):
    ntasks = len(tasks)
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(ntasks)])
    self.ps = [Process(target=worker, args=(work_remote, remote, task))
               for (work_remote, remote, task) in zip(self.work_remotes, self.remotes, tasks)]
  
    for p in self.ps:
      p.daemon = True
      p.start()
    
    for remote in self.work_remotes:
      remote.close()

    self.remotes[0].send(('get_spaces', None))
    self.observation_space, self.action_space = self.remotes[0].recv()

  def step_async(self, actions):
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', action))
    self.waiting = True

  def step_wait(self):
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False

  def reset(self):
    for remote in self.remotes:
      remote.send(('reset', None))
    return np.stack([remote.recv() for remote in self.remotes])

  def close(self):
    if self.closed:
      return
    if self.waiting:
      for remote in self.remotes:
        remote.recv()
