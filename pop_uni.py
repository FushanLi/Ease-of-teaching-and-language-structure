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
args['trainIters'] = 300000 
args['sLearnRate'] = 0.001 
args['rLearnRate'] = 0.001 

args['resetNum'] = 50 
args['resetIters'] = args['trainIters'] // args['resetNum']  # life of a receiver, iters of one population
args['deterResetNums'] = 30
args['deterResetIter'] = 1000

# population of receivers training
args['population'] = True
args['renewIters'] = args['resetIters'] // args['receiverNum']
args['renewNum'] = args['trainIters'] // args['renewIters']

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
team = popGuessGame(args)
# get data
data = Dataset(args['numColors'], args['numShapes'], args['attrSize'])
train_np = data.getTrain()
util = utility.Utility(args, data)

sloss_l = np.zeros(args['trainIters'])
pop_rloss_l = np.zeros(args['trainIters'])  # record mean rloss
pop_acc_l = np.zeros(args['trainIters'])  # use mean rewards to calculate accuracy

entropy_l = np.zeros(args['trainIters'])

# easy-to-teach evaluation
evalAcc_l = np.zeros((args['resetNum'] // 10, args['deterResetNums'], args['deterResetIter']))

dTopo = np.zeros(args['resetNum']+1)
dEntropy = np.zeros(args['resetNum']+1)

def eval_teach_speed(eval_ind, data, team):
    # make deterministic, just when training let sender speaks deterministic language
    print('Evaluate the teaching speed after reset for ' + str(10 * (eval_ind+1)) + ' receivers')
    rbot, rOptimizer = team.evalReceiver()

    for i in range(args['deterResetNums']):
        # evaluate before reset
        print('Reset the ' + str(i+1) + 'th receiver with deterministic language')  # start from 1
        rOptimizer = team.resetReceiver(rbot, rOptimizer)

        for j in range(args['deterResetIter']):
            candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'])
            rloss, rewards = team.evalForward(rbot, targets, candidates)  # speak in evaluate mode
            team.evalbackward(rloss, rOptimizer)  

            evalAcc_l[eval_ind][i][j] = rewards.sum().item() / args['batchSize'] * 100  # reward +1 0

            # print intermediate results during training
            if j == 0 or (j + 1) % 100 == 0:
                record = 'Iteration ' + str(i * args['deterResetIter'] + j + 1) \
                         + ' Training accuracy ' + str(np.round(evalAcc_l[eval_ind][i][j], decimals=2)) + '%\n'
                print(record)

with torch.no_grad():
    dTopo[0], dEntropy[0], prevLangD = util.get_sender_language(team, neural=True)  # evaluate all group performance

for i in range(args['trainIters']):
    candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'])  # same data for all receivers
    sloss, pop_rloss_l[i], rloss_list, rewards_mean, rewards_list, entropy, _, _ = team.popForward(targets, candidates, evaluate=False)

    pop_acc_l[i] = rewards_mean.sum().item() / args['batchSize'] * 100  # reward +1 0
    sloss_l[i] = sloss
    entropy_l[i] = entropy

    team.popBackward(sloss, rloss_list)

    if i % 100 == 0:
        record = 'Iteration ' + str(i) \
                 + ' Sender loss ' + str(np.round(sloss_l[i], decimals=4)) \
                 + ' Recever loss ' + str(np.round(pop_rloss_l[i], decimals=4)) \
                 + ' Training accuracy ' + str(np.round(pop_acc_l[i], decimals=2)) + '%\n'
        print(record)

    # after one generation 6k
    if i != 0 and i % args['resetIters'] == 0:
        with torch.no_grad():
            ind = i // args['resetIters']
            dTopo[ind], dEntropy[ind], curLangD = util.get_sender_language(team, neural=True)

        if ind % 10 == 0:
            team.freezeSender()
            eval_teach_speed(ind // 10 - 1, data, team)
            team.defreezeSender()
    # 600
    if i != 0 and i % args['renewIters'] == 0:
        team.renewOneReceiver()


print('After training for ' + str(args['trainIters']) + ' iterations')
with torch.no_grad():
    dTopo[-1], dEntropy[-1], langD = util.get_sender_language(team, neural=True)
    np.save(args['fname'] + '/langDict', langD)

# speed of teaching the language to a new listener after determinized
# make params untrainable, testing if sender is not learning

team.freezeSender()
eval_teach_speed(args['resetNum'] // 10 - 1, data, team)

np.save(args['fname'] + '/sloss', sloss_l)
np.save(args['fname'] + '/pop_rloss', pop_rloss_l)
np.save(args['fname'] + '/pop_trainAcc', pop_acc_l)
np.save(args['fname'] + '/entropy', entropy_l)
np.save(args['fname'] + '/dTopo', dTopo)
np.save(args['fname'] + '/dEntropy', dEntropy)
np.save(args['fname'] + '/evalAcc', evalAcc_l)
