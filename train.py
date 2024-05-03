# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

import os

from tensorboardX import SummaryWriter

class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400 # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 20000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 2
        self.epochs = 6  # num of train_steps for each update, origin: 5, try: 3
        self.kl_targ = 0.02
        self.check_freq = 20
        self.game_batch_num = 4000

        self.eval_games = 10 
        self.best_win_ratio = 0.525
        self.five_to_five_cnt = 0

        self.use_pretrained = False  #decide to use model trained before or not
        # init_model = f"./models_{self.board_width}_{self.board_height}_{self.n_in_row}_me/policyNet_init.model"
        if self.use_pretrained and os.path.isfile(init_model):
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model, use_gpu=True)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,use_gpu=True)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        self.episode_len = 0
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len += len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
        self.episode_len /= n_games

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        loss = 0
        entropy = 0
        for i in range(self.epochs):
            loss_one, entropy_one = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            loss += loss_one
            entropy += entropy_one
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        loss /= self.epochs
        entropy /= self.epochs

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # explained_var_old = (1 -
        #                      np.var(np.array(winner_batch) - old_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        # explained_var_new = (1 -
        #                      np.var(np.array(winner_batch) - new_v.flatten()) /
        #                      np.var(np.array(winner_batch)))
        # print(("kl:{:.5f},"
        #        "lr_multiplier:{:.3f},"
        #        "loss:{},"
        #        "entropy:{},"
        #        "explained_var_old:{:.3f},"
        #        "explained_var_new:{:.3f}"
        #        ).format(kl,
        #                 self.lr_multiplier,
        #                 loss,
        #                 entropy,
        #                 explained_var_old,
        #                 explained_var_new))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{:.6f},"
               "entropy:{:.6f}"
               ).format(kl, self.lr_multiplier, loss, entropy))
        
        return kl, loss

    def policy_evaluate(self, n_games):
        """
        Evaluate the current trained policy by playing against the best_saved model (RL就是一个自我学习成长的, 所以, 和自己比有进步就可以了)!!!!!!
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        
        before_best_model = f"./models_{self.board_width}_{self.board_height}_{self.n_in_row}_me/best_policy.model"
        if not os.path.isfile(before_best_model):   #如果刚开始还没有这个文件的话, 就直接返回一个稍微比阈值大的值, 从而选择保存当前模型
            return self.best_win_ratio + 0.01

        before_policy_value_net = PolicyValueNet(self.board_width,
                                                 self.board_height,
                                                 model_file=before_best_model,use_gpu=True)
        before_best_player = MCTSPlayer(before_policy_value_net.policy_value_fn,
                                        c_puct=self.c_puct,
                                        n_playout=self.n_playout)
        
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                before_best_player,
                                start_player=i % 2,
                                is_shown=0)
            win_cnt[winner] += 1

        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("Compete with before model --- win: {}, lose: {}, tie:{}".format(
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
        
    # def policy_evaluate(self, n_games=10):
    #     """
    #     Evaluate the trained policy by playing against the pure MCTS player
    #     Note: this is only for monitoring the progress of training
    #     """
    #     current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
    #                                      c_puct=self.c_puct,
    #                                      n_playout=self.n_playout)
    #     pure_mcts_player = MCTS_Pure(c_puct=5,
    #                                  n_playout=self.pure_mcts_playout_num)
    #     win_cnt = defaultdict(int)  #这个字典多了一个功能, 就是创建一个key时自动赋值为0
    #     for i in range(n_games):
    #         winner = self.game.start_play(current_mcts_player,
    #                                       pure_mcts_player,
    #                                       start_player=i % 2,
    #                                       is_shown=0)
    #         win_cnt[winner] += 1
        
    #     win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games    # current_mcts_player是player1, 因此这里的win_ratio指的是current_player的胜率
    #     print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
    #             self.pure_mcts_playout_num,
    #             win_cnt[1], win_cnt[2], win_cnt[-1]))
    #     return win_ratio
    
    def run(self):
        """run the training pipeline"""
        try:
            writer = SummaryWriter(f"./runs/{self.board_width}_{self.board_height}_{self.n_in_row}_leaf_damping")

            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    kl, loss = self.policy_update()
                    writer.add_scalar('Loss', loss, i+1)
                    writer.add_scalar('KL', kl, i+1)
                    writer.add_scalar('LR_multiplier', self.lr_multiplier, i+1)
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate(n_games=self.eval_games)
                    self.policy_value_net.save_model(f"./current_policy.model")
                    # if win_ratio > self.best_win_ratio:
                    #     print("New best policy!!!!!!!!")
                    #     self.best_win_ratio = win_ratio
                    #     # update the best_policy
                    #     self.policy_value_net.save_model(f"./models_{self.board_width}_{self.board_height}_{self.n_in_row}_me/best_policy.model")
                    #     if (self.best_win_ratio == 1.0 and
                    #             self.pure_mcts_playout_num < 5000):
                    #         self.pure_mcts_playout_num += 1000
                    #         self.best_win_ratio = 0.0
                    if win_ratio >= self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.policy_value_net.save_model(f"./models_{self.board_width}_{self.board_height}_{self.n_in_row}_me/best_policy.model")
                        self.five_to_five_cnt = 0
                    if win_ratio == 0.5:
                        self.five_to_five_cnt += 1
                        if self.five_to_five_cnt == 3:
                            self.eval_games += 2
                            # self.best_win_ratio = max(0.51, self.best_win_ratio - 0.01)
                            self.n_playout += 100
                            self.buffer_size += 10000
                            self.batch_size += 32
                            self.data_buffer = deque(maxlen=self.buffer_size)

                            self.five_to_five_cnt = 0
  
        except KeyboardInterrupt:
            print('\n\rquit')
        
        finally:
            writer.close()


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
