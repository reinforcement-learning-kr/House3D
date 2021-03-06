from House3D import Environment, objrender, load_config
from House3D.roomnav import RoomNavTask
from model import A3C_LSTM_GA
from collections import deque
from tensorboardX import SummaryWriter
import numpy as np
import args


from torch.autograd import Variable
import torch


EPISODE=100000

api = objrender.RenderAPI(
    w=400, h=300, device=0)
cfg = load_config('config.json')

targets = ['bedroom', 'kitchen', 'bathroom', 'dining_room', 'living_room']
actions = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

houses = ['00065ecbdd7300d35ef4328ffe871505',
    'cf57359cd8603c3d9149445fb4040d90', '31966fdc9f9c87862989fae8ae906295',
    '7995c2a93311717a3a9c48d789563590', '8b8c1994f3286bfc444a7527ffacde86',
    '32e53679b33adfcc5a5660b8c758cc96',
    '492c5839f8a534a673c92912aedc7b63',
    'e3ae3f7b32cf99b29d3c8681ec3be321',
    '1dba3a1039c6ec1a3c141a1cb0ad0757',
    '5f3f959c7b3e6f091898caa8e828f110']

## tensorboard
writer = SummaryWriter(log_dir="./checkpoints/log")

def cuda(x):
  if torch.cuda.is_available():
    return x.cuda()
  else:
    return x


def main():
  ## make environment and task
  env = Environment(api, np.random.choice(houses, 1)[0], cfg)
  task = RoomNavTask(env, hardness=0.6, discrete_action=True)

  ## make gated attention network
  net = cuda(A3C_LSTM_GA(len(targets)))

  succ = deque(maxlen=500)
  traj = []
  total_step = 0

  ## main loop for interact with environment
  for i in range(EPISODE):
    ## initialize task
    step, total_rew, good = 0, 0, 0
    obs = task.reset()
    target = task.info['target_room']
  
    target = [1 if targets[i] == 1 else 0 for i in range(len(targets))]
    target = cuda(Variable(torch.FloatTensor(target)))
  
    ## initialize hidden state
    hx = cuda(Variable(torch.zeros(1, 256)))
    cx = cuda(Variable(torch.zeros(1, 256)))

    while True:
      obs = cuda(Variable(torch.FloatTensor(obs)))
      value, prob, act, hx, cx = net.action(obs.permute(2, 0, 1)[None], 
                                             target, hx, cx)
      next_obs, rew, done, info = task.step(actions[act])
      total_rew += rew
      
      ## append data
      traj.append([act, rew, done, prob, value])
      
      ## train!
      if len(traj) == args.max_steps:
        train(traj, total_step)
        traj = []

      step += 1
      total_step += 1

      obs = next_obs
      
      if done or step > 150:
        if done:
          good = 1
        succ.append(good)
        
        ## print logs
        print("\n+++++++++++++ status ++++++++++++")
        print("Episode {:4d}, Reward: {:2.3f}".format(i+1, total_rew))
        print("Target {}".format(task.info['target_room']))
        print("Success rate {:3.2f}%".format(np.sum(succ)/len(succ)*100))
        break

        ## tensorboard summary writer
        writer.add_scalar("Success rate", np.sum(succ)/len(succ), i+1)
        writer.add_scalar("Reward", total_rew, i+1)


def train(traj, step):
  acts, rews, dones, probs, values = list(zip(*traj))
  
  acts = np.eye(10)[[acts]]
  rews = cuda(torch.FloatTensor(rews))
  dones = 1 - np.array(dones)
  probs = torch.stack(probs)
  values = torch.stack(values)
  

  ## initialize R
  value_loss = 0
  policy_loss = 0
  R = torch.zeros(1)
  if dones[-1]:
    R = values[-1]
  
  ## GAE
  for i in reversed(range(len(dones))):
    R = args.gamma * R + rews[i]
    advantages = R - values[i]
    
    value_loss += 0


if __name__ == "__main__":
  main()
