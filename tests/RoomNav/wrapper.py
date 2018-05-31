def worker(remote, parent_remote, env):
  parernt_remote.close()


class TaskWrapper:
  def __init__(self, tasks):
    ntasks = len(tasks)
    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(ntasks)])
    self.ps = [Process(target=worker, args=(work_remote, remote, task))
               for (work_remote, remote, task) in zip(self.work_remotes, self.remotes, tasks)]
  


