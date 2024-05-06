from collections import defaultdict

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

def run():
    n = 5
    width, height = 9, 9
    N_fights = 10
    
    player1_file = f"./models_{width}_{height}_{n}_me/best_policy(leafDamp).model"
    player2_file = f"./models_{width}_{height}_{n}_me/best_policy(non-leaf).model"

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### AI VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow
        best_policy1 = PolicyValueNet(width, height, model_file = player1_file, use_gpu=True)
        mcts_player1 = MCTSPlayer(best_policy1.policy_value_fn, c_puct=5, n_playout=800)     

        best_policy2 = PolicyValueNet(width, height, model_file = player2_file, use_gpu=True)
        mcts_player2 = MCTSPlayer(best_policy2.policy_value_fn, c_puct=5, n_playout=800)

        win_cnt = defaultdict(int)
        for i in range(N_fights):
            winner = game.start_play(mcts_player1, mcts_player2, start_player=i % 2, is_shown=0)
            if winner == 1:
                print(f"round {i+1}: player 1 win!")
            elif winner == 2:
                print(f"round {i+1}: player 2 win!")
            else:
                print(f"round {i+1}: Tie!")
            win_cnt[winner] += 1

        print("player1 win: {}, player2 win: {}, tie:{}".format(
                win_cnt[1], win_cnt[2], win_cnt[-1]))

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
