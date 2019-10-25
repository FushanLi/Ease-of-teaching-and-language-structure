from __future__ import print_function
from __future__ import division
import torch
import torch.optim as optim
import models
import numpy as np


class popGuessGame:
    def __init__(self, args):
        self.args = args
        for key, value in args.items():
            setattr(self, key, value)

        self.sbot = models.Sender(args).to(self.device)
        print('sbot', self.sbot)
        self.sOptimizer = optim.Adam(self.sbot.parameters(), lr=self.sLearnRate)

        self.rbotGroup = [models.Receiver(args).to(self.device) for _ in range(self.receiverNum)]
        print('group size', len(self.rbotGroup))
        self.rOptimizerGroup = [optim.Adam(r.parameters(), lr=self.rLearnRate) for r in self.rbotGroup]
        self.oldest = 0

    def popForward(self, targets, candidates, evaluate=False, stochastic=True):
        targetsTensor = torch.from_numpy(targets).to(self.device)

        batch = targetsTensor.size()[0]
        self.sbot.init_hidden(batch)
        for r in self.rbotGroup:
            r.init_hidden(batch)

        if evaluate:
            self.sbot.eval() 
            for r in self.rbotGroup:
                r.eval()
        else:
            self.sbot.train()
            for r in self.rbotGroup:
                r.train()

        m, speak_log_probs, speak_p_log_p, evaluate_probs = self.sbot.speak(targetsTensor, stochastic)

        for r in self.rbotGroup:
            r.listen(m)
        candidatesTensor = torch.from_numpy(candidates).to(self.device)

        pred_log_probs_list = []
        pred_p_log_p_list = []
        rewards_list = []
        for r in self.rbotGroup:
            p_action, pred_log_probs, pred_p_log_p, pred_probs = r.predict(candidatesTensor, stochastic)

            pred_log_probs_list.append(pred_log_probs)
            pred_p_log_p_list.append(pred_p_log_p)

            predicts = candidatesTensor[np.arange(batch), p_action, :]
            mattersIndex = self.numColors + self.numShapes
            tsum = torch.LongTensor(batch).fill_(mattersIndex).to(self.device)
            psum = torch.eq(predicts[:,0:mattersIndex], targetsTensor[:, 0:mattersIndex]).sum(1) # true if predicts and targets match at every index

            rewards = torch.eq((tsum), psum).float()
            rewards_list.append(rewards)

        rewards_mean = torch.mean(torch.stack(rewards_list, dim=0), dim=0)
        sloss = -rewards_mean * speak_log_probs + self.slambda * speak_p_log_p
        sloss = torch.sum(sloss) / batch

        rloss_list = []
        for i in range(len(self.rbotGroup)):
            rloss = -rewards_list[i] * pred_log_probs_list[i] + self.rlambda * pred_p_log_p_list[i]
            rloss = torch.sum(rloss) / batch
            rloss_list.append(rloss)
        rloss_mean = torch.mean(torch.stack(rloss_list, dim=0), dim=0)

        batch_entropy = -torch.sum(speak_p_log_p) / batch

        return sloss, rloss_mean, rloss_list, rewards_mean, rewards_list, batch_entropy, m, evaluate_probs

    def popBackward(self, sloss, rloss_list):

        self.sOptimizer.zero_grad()
        sloss.backward()
        self.sOptimizer.step()

        for i in range(self.receiverNum):
            self.rOptimizerGroup[i].zero_grad()
            rloss_list[i].backward()
            self.rOptimizerGroup[i].step()

    def renewOneReceiver(self):
        self.rbotGroup.pop(self.oldest)
        self.rOptimizerGroup.pop(self.oldest)

        self.rbotGroup.insert(self.oldest, models.Receiver(self.args).to(self.device))
        self.rOptimizerGroup.insert(self.oldest, optim.Adam(self.rbotGroup[self.oldest].parameters(), lr=self.rLearnRate))
        print('Kickoff ', str(self.oldest), ' th receiver and get a newbie')

        if self.oldest < self.receiverNum - 1: 
            self.oldest += 1
        else:
            self.oldest = 0

        # update sender optimizer
        self.sOptimizer = optim.Adam(self.sbot.parameters(), lr=self.sLearnRate)

    def senderForward(self, targets, neural):
        targetsTensor = torch.from_numpy(targets).to(self.device) 
        batch = targetsTensor.size()[0]

        if neural:
            self.sbot.init_hidden(batch)
            self.sbot.eval()
        m, _, p_log_p, speak_probs = self.sbot.speak(targetsTensor, stochastic=False)
        deter_entropy = -torch.sum(p_log_p) / batch

        return m, deter_entropy, speak_probs

    def freezeSender(self):
        for param in self.sbot.parameters():
            param.requires_grad = False
        print('\nSender parameters are freezed now')

    def defreezeSender(self):
        for param in self.sbot.parameters():
            param.requires_grad = True
        print('\nSender parameters are trainable now')

    def evalReceiver(self):
        rbot = models.Receiver(self.args).to(self.device)
        rOptimizer = optim.Adam(rbot.parameters(), lr=self.rLearnRate)
        return rbot, rOptimizer

    def resetReceiver(self, rbot, rOptimizer):
        rbot.attr2embed.reset_parameters()
        rbot.listenlstm.reset_parameters()
        rbot.hidden2embed.reset_parameters()
        print('\nParameters in the Receiver are reset')

        rOptimizer = optim.Adam(rbot.parameters(), lr=self.rLearnRate)
        print('Reinitialize receiver optimizer')
        return rOptimizer

    def popResetReceivers(self):
        self.rOptimizerGroup = []
        for ind, rbot in enumerate(self.rbotGroup):
            rbot.attr2embed.reset_parameters()
            rbot.listenlstm.reset_parameters()
            rbot.hidden2embed.reset_parameters()
            print('\nParameters in the Receiver ' + str(ind) + ' are reset')

            rOptimizer = optim.Adam(rbot.parameters(), lr=self.rLearnRate)
            self.rOptimizerGroup.append(rOptimizer)

        # update sender optimizer
        self.sOptimizer = optim.Adam(self.sbot.parameters(), lr=self.sLearnRate)
        print('Reinitialize sender optimizer')

    def evalForward(self, rbot, targets, candidates):
        targetsTensor = torch.from_numpy(targets).to(self.device) 
        batch = targetsTensor.size()[0]

        self.sbot.init_hidden(batch)
        rbot.init_hidden(batch)

        self.sbot.eval()  
        rbot.train()

        m, speak_log_probs, speak_p_log_p, evaluate_probs = self.sbot.speak(targetsTensor, stochastic=False)

        rbot.listen(m)

        candidatesTensor = torch.from_numpy(candidates).to(self.device)
        p_action, pred_log_probs, pred_p_log_p, pred_probs = rbot.predict(candidatesTensor, stochastic=False)

        predicts = candidatesTensor[np.arange(batch), p_action, :]
        mattersIndex = self.numColors + self.numShapes
        tsum = torch.LongTensor(batch).fill_(mattersIndex).to(self.device)
        psum = torch.eq(predicts[:, 0:mattersIndex], targetsTensor[:, 0:mattersIndex]).sum(1)  # true if predicts and targets match at every index

        rewards = torch.eq(tsum, psum).float()

        rloss = -rewards * pred_log_probs + 0.1 * pred_p_log_p
        rloss = torch.sum(rloss) / batch

        return rloss, rewards

    def evalbackward(self, rloss, rOptimizer):
        rOptimizer.zero_grad()
        rloss.backward()
        rOptimizer.step()

