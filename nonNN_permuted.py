"""
Teaching an artificial language to listeners
"""

from dataGenerator import Dataset
import utility
from game import GuessGame
import models
import torch
import numpy as np
import parser
import os

args = parser.parse()
args['device'] = torch.device("cuda:" + str(args['gpu']) if torch.cuda.is_available() else "cpu")
if not os.path.exists(args['fname']):
    os.makedirs(args['fname'])

# dataset hyperparameters
args['numColors'] = 8 
args['numShapes'] = 4 
args['attrSize'] = args['numColors'] + args['numShapes']  # colors + shapes

# game settings
args['vocabSize'] = 8
args['messageLen'] = 2
args['distractNum'] = 5  # including targets
args['topk'] = 3

# model hyperparameters
args['hiddenSize'] = 100 

# training hyperparameters
args['batchSize'] = 100  # total train data = batchSize * numIters
args['sLearnRate'] = 0.001  
args['rLearnRate'] = 0.001 
args['slambda'] = 0.1  # regularizer, the larger the more exploration
args['rlambda'] = 0.05

args['trainIters'] = 1000
args['deterResetNums'] = 30

# get data
data = Dataset(args['numColors'], args['numShapes'], args['attrSize'])
train_np = data.getTrain()
util = utility.Utility(args, data)

# generate game settings
team = GuessGame(args)
team.sbot = models.overlapPermutedSender(args)
team.sOptimizer = None
sOpt = False

# get sender language
dTopo, _, langD = util.get_sender_language(team, neural=False) # dTopo should be the same for each reset
np.save(args['fname'] + '/langDict', langD)

sloss_l = np.zeros(args['trainIters'] * args['deterResetNums'])
rloss_l = np.zeros(args['trainIters'] * args['deterResetNums'])
trainAccuracy_l = np.zeros(args['trainIters'] * args['deterResetNums'])

for i in range(args['trainIters'] * args['deterResetNums']):
    candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'])
    sloss, rloss, message, rewards, _, _, _ = team.forward(targets, candidates, False, sOpt, True, False)
    team.backward(sloss, rloss, sOpt)

    sloss_l[i] = sloss
    rloss_l[i] = rloss
    trainAccuracy_l[i] = rewards.sum().item() / args['batchSize'] * 100  # reward +1 0

    # print intermediate results during training
    if i % 100 == 0:
        record = 'Iteration ' + str(i) \
                 + ' Sender loss ' + str(np.round(sloss_l[i], decimals=4)) \
                 + ' Recever loss ' + str(np.round(rloss_l[i], decimals=4)) \
                 + ' Training accuracy ' + str(np.round(trainAccuracy_l[i], decimals=2)) + '%\n'
        print(record)

    if i != 0 and i % args['trainIters'] == 0:
        # evaluate before reset
        print('Reset the ' + str(i // args['trainIters']) + 'th receiver with deterministic language')  # start from 1
        team.resetReceiver(sOpt)

np.save(args['fname'] + '/sloss', sloss_l)
np.save(args['fname'] + '/rloss', rloss_l)
np.save(args['fname'] + '/trainAcc', trainAccuracy_l)

np.save(args['fname'] + '/dTopo', dTopo)
