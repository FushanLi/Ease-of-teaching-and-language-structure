"""
Evaluating teaching speed of language during resets

"""
from __future__ import print_function
from __future__ import division
import torch
import parser
import os
import random
import sys
args = parser.parse()  # parsed argument from CLI
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

# training hyperparameters
args['batchSize'] = 100  # total train data = batchSize * numIters
args['sLearnRate'] = 0.001  
args['rLearnRate'] = 0.001  

args['trainIters'] = 300000 # training
args['resetNum'] = 50  
args['resetIter'] = args['trainIters'] // args['resetNum']  # life of a receiver: 6K
args['deterResetNums'] = 30
args['deterResetIter'] = 1000

# population of receivers training
args['population'] = False

# model hyperparameters
args['hiddenSize'] = 100 

print(args)

from dataGenerator import Dataset
import utility
import numpy as np
if args['population']:
  from popgame import popGuessGame
else:
  from game import GuessGame

torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
np.random.seed(args['seed'])
random.seed(args['seed'])

torch.backends.cudnn.deterministic=True

# @title Train
team = GuessGame(args)
# get data
data = Dataset(args['numColors'], args['numShapes'], args['attrSize'])
train_np = data.getTrain()
util = utility.Utility(args, data)

sloss_l = np.zeros(args['trainIters'])
rloss_l = np.zeros(args['trainIters'])
trainAccuracy_l = np.zeros(args['trainIters'])
entropy_l = np.zeros(args['trainIters'])

# easy-to-teach evaluation
evalAcc_l = np.zeros((args['resetNum'] // 10, args['deterResetNums'], args['deterResetIter']))

dTopo = np.zeros(args['resetNum']+1)
dEntropy = np.zeros(args['resetNum']+1)

def eval_teach_speed(eval_ind, data, team):
    # make deterministic, just when training let sender speaks deterministic language
    print('Evaluate the teaching speed after reset for ' + str(10 * (eval_ind+1)) + ' receivers')

    for i in range(args['deterResetNums']):
        # evaluate before reset
        print('Reset the ' + str(i+1) + 'th receiver with deterministic language')  # start from 1
        team.resetReceiver(sOpt=False)

        for j in range(args['deterResetIter']):
            candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'])
            sloss, rloss, message, rewards, _, _, _ = team.forward(targets, candidates, evaluate=False, sOpt=True, rOpt=True,
                                                                stochastic=False)  # speak in evaluate mode
            team.backward(sloss, rloss, sOpt=False)  

            evalAcc_l[eval_ind][i][j] = rewards.sum().item() / args['batchSize'] * 100  # reward +1 0

            # print intermediate results during training
            if j == 0 or (j + 1) % 100 == 0:
                record = 'Iteration ' + str(i * args['deterResetIter'] + j + 1) \
                         + ' Training accuracy ' + str(np.round(evalAcc_l[eval_ind][i][j], decimals=2)) + '%\n'
                print(record)

with torch.no_grad():
    dTopo[0], dEntropy[0], prevLangD = util.get_sender_language(team, neural=True)  # evaluate all group performance

for i in range(args['trainIters']):
    candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'])
    sloss, rloss, message, rewards, entropy, _, _ = team.forward(targets, candidates, False, True, True, stochastic=True)
    team.backward(sloss, rloss)

    sloss_l[i] = sloss
    rloss_l[i] = rloss
    trainAccuracy_l[i] = rewards.sum().item() / args['batchSize'] * 100  # reward +1 0
    entropy_l[i] = entropy

    # print intermediate results during training
    if i % 100 == 0:
        record = 'Iteration ' + str(i) \
                 + ' Sender loss ' + str(np.round(sloss_l[i], decimals=4)) \
                 + ' Recever loss ' + str(np.round(rloss_l[i], decimals=4)) \
                 + ' Training accuracy ' + str(np.round(trainAccuracy_l[i], decimals=2)) + '%\n'
        print(record)

    if i != 0 and i % args['resetIter'] == 0:
        # evaluate before reset
        print('Before reset: ')
        print('For the ' + str(i // args['resetIter']) + 'th receiver')  # start from 1
        with torch.no_grad():
            ind = i // args['resetIter']
            dTopo[ind], dEntropy[ind], curLangD = util.get_sender_language(team, neural=True) # calculate topo similarity before each reset
        if ind % 10 == 0:
            team.freezeSender()
            eval_teach_speed(ind // 10 - 1, data, team)
            team.defreezeSender()

        team.resetReceiver()

print('After training for ' + str(args['trainIters']) + ' iterations')
with torch.no_grad():
    dTopo[-1], dEntropy[-1], langD = util.get_sender_language(team, neural=True) # evaluate all group performance
    np.save(args['fname'] + '/langDict', langD)

# speed of teaching the language to a new listener after determinized
# make params untrainable, testing if sender is not learning
team.freezeSender()
eval_teach_speed(args['resetNum'] // 10 - 1, data, team)

np.save(args['fname'] + '/sloss', sloss_l)
np.save(args['fname'] + '/rloss', rloss_l)
np.save(args['fname'] + '/trainAcc', trainAccuracy_l)
np.save(args['fname'] + '/entropy', entropy_l)
np.save(args['fname'] + '/dTopo', dTopo)
np.save(args['fname'] + '/dEntropy', dEntropy)
np.save(args['fname'] + '/evalAcc', evalAcc_l)
