# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-

"""人机对战"""

"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

import numpy as np人机对战和AI演示# @Time : 2025/11/9# @Time : 2025/11/8

from model.game import DotsAndBoxesGame

from model.model import DotsAndBoxesNet"""

from model.mcts import MCTS

# @Author : ironhxs# @Author : ironhxs



def play():import torch

    game = DotsAndBoxesGame(5, 5)

    nnet = DotsAndBoxesNet(game)import numpy as np# @File : play.py# @File : play.py

    checkpoint = torch.load('checkpoints/latest.pth')

    nnet.load_state_dict(checkpoint['state_dict'])from game import DotsAndBoxesGame

    if torch.cuda.is_available():

        nnet.cuda()from model import DotsAndBoxesNet

    nnet.eval()

    from mcts import MCTS

    args = {'num_simulations': 100, 'cpuct': 1.0, 'dirichlet_alpha': 0.3, 'dirichlet_epsilon': 0.0, 'temp_threshold': 0}

    mcts = MCTS(game, nnet, args)import torch"""

    state = game.get_initial_state()

    human_player = 0

    

    print("\n" + "=" * 60)def play_human_vs_ai():import numpy as np人机对战模块

    print("点格棋 - 人类 vs AI")

    print("=" * 60)    game = DotsAndBoxesGame(5, 5)

    

    while not game.is_terminal(state):    nnet = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)from game import DotsAndBoxesGame"""

        game.display(state)

        current_player = game.get_current_player(state)    

        

        if current_player == human_player:    checkpoint = torch.load('checkpoints/latest.pth')from model import DotsAndBoxesNet

            valid_moves = game.get_valid_moves(state)

            valid_actions = [i for i, v in enumerate(valid_moves) if v == 1]    nnet.load_state_dict(checkpoint['state_dict'])

            print(f"合法动作: {valid_actions}")

                from mcts import MCTSimport torch

            while True:

                try:    if torch.cuda.is_available():

                    action = int(input("输入动作: "))

                    if action in valid_actions:        nnet.cuda()from .game import DotsAndBoxesGame

                        break

                    print("无效动作!")    

                except:

                    print("请输入数字!")    nnet.eval()def play_human_vs_ai():from .model import DotsAndBoxesNNet

        else:

            print("AI思考中...")    

            pi = mcts.get_action_prob(state, temp=0)

            action = np.argmax(pi)    args = {    """人类 vs AI 对战"""from .evaluate import AIPlayer

            print(f"AI选择: {action}")

                'num_simulations': 100,

        state = game.get_next_state(state, action)

            'cpuct': 1.0,    from . import config

    game.display(state)

    result = game.get_game_result(state, human_player)        'dirichlet_alpha': 0.3,

    print("🎉 你赢了!" if result == 1 else "😢 AI赢了!" if result == -1 else "🤝 平局!")

        'dirichlet_epsilon': 0.0,    # 游戏配置



if __name__ == '__main__':        'temp_threshold': 0,

    play()

    }    game = DotsAndBoxesGame(5, 5)class HumanPlayer:

    

    mcts = MCTS(game, nnet, args)        """人类玩家"""

    state = game.get_initial_state()

    human_player = 0    # 加载训练好的模型    def __init__(self, game):

    

    print("\n" + "=" * 60)    nnet = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)        self.game = game

    print("点格棋 - 人类 vs AI")

    print("=" * 60)        

    print(f"你是玩家 {human_player}")

    print("输入边的编号来放置边")    checkpoint = torch.load('checkpoints/latest.pth')    def get_action(self, state, temp=0):

    print("=" * 60 + "\n")

        nnet.load_state_dict(checkpoint['state_dict'])        """

    while not game.is_terminal(state):

        game.display(state)            从用户输入获取动作

        current_player = game.get_current_player(state)

            if torch.cuda.is_available():        

        if current_player == human_player:

            valid_moves = game.get_valid_moves(state)        nnet.cuda()        Args:

            valid_actions = [i for i, v in enumerate(valid_moves) if v == 1]

            print(f"合法动作: {valid_actions}")                state: 当前游戏状态

            

            while True:    nnet.eval()            temp: 占位符，保持接口一致

                try:

                    action = int(input("输入动作编号: "))                

                    if action in valid_actions:

                        break    # MCTS配置        Returns:

                    print("无效动作，请重新输入!")

                except:    args = {            action: 用户选择的动作

                    print("请输入数字!")

        else:        'num_simulations': 100,        """

            print("AI思考中...")

            pi = mcts.get_action_prob(state, temp=0)        'cpuct': 1.0,        valids = self.game.get_valid_moves(state)

            action = np.argmax(pi)

            print(f"AI选择动作: {action}")        'dirichlet_alpha': 0.3,        valid_actions = [i for i, v in enumerate(valids) if v == 1]

        

        state = game.get_next_state(state, action)        'dirichlet_epsilon': 0.0,  # 对战时不添加噪声        

    

    game.display(state)        'temp_threshold': 0,        print("\nLegal actions:")

    result = game.get_game_result(state, human_player)

    if result == 1:    }        for i, action in enumerate(valid_actions[:10]):  # 只显示前10个

        print("🎉 你赢了!")

    elif result == -1:                print(f"  {i}: Action {action}")

        print("😢 AI赢了!")

    else:    mcts = MCTS(game, nnet, args)        if len(valid_actions) > 10:

        print("🤝 平局!")

                print(f"  ... and {len(valid_actions) - 10} more")



def play_ai_vs_ai():    # 开始游戏        

    game = DotsAndBoxesGame(5, 5)

    nnet = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)    state = game.get_initial_state()        while True:

    

    checkpoint = torch.load('checkpoints/latest.pth')    human_player = 0  # 人类是玩家0            try:

    nnet.load_state_dict(checkpoint['state_dict'])

                        action_idx = int(input("\nEnter action index (or action number): "))

    if torch.cuda.is_available():

        nnet.cuda()    print("\n" + "=" * 60)                

    

    nnet.eval()    print("点格棋 - 人类 vs AI")                # 尝试作为索引

    

    args = {    print("=" * 60)                if 0 <= action_idx < len(valid_actions):

        'num_simulations': 100,

        'cpuct': 1.0,    print(f"你是玩家 {human_player}")                    return valid_actions[action_idx]

        'dirichlet_alpha': 0.3,

        'dirichlet_epsilon': 0.0,    print("输入边的编号来放置边")                

        'temp_threshold': 0,

    }    print("=" * 60 + "\n")                # 尝试作为动作编号

    

    mcts = MCTS(game, nnet, args)                    if action_idx in valid_actions:

    state = game.get_initial_state()

        while not game.is_terminal(state):                    return action_idx

    print("\n" + "=" * 60)

    print("点格棋 - AI vs AI 演示")        game.display(state)                

    print("=" * 60 + "\n")

                            print(f"Invalid action. Please choose from 0-{len(valid_actions)-1} or a valid action number.")

    move_count = 0

    while not game.is_terminal(state):        current_player = game.get_current_player(state)            except ValueError:

        move_count += 1

        print(f"\n--- 第 {move_count} 步 ---")                        print("Invalid input. Please enter a number.")

        

        pi = mcts.get_action_prob(state, temp=0)        if current_player == human_player:            except KeyboardInterrupt:

        action = np.argmax(pi)

                    # 人类回合                print("\nGame interrupted.")

        print(f"玩家 {game.get_current_player(state)} 选择动作: {action}")

        state = game.get_next_state(state, action)            valid_moves = game.get_valid_moves(state)                return None

        

        game.display(state)            valid_actions = [i for i, v in enumerate(valid_moves) if v == 1]    

        input("按回车继续...")

                    def reset(self):

    returns = state.returns()

    print(f"\n游戏结束! 玩家0: {returns[0]}, 玩家1: {returns[1]}")            print(f"合法动作: {valid_actions}")        """占位符"""



                    pass

if __name__ == '__main__':

    import sys            while True:

    

    if len(sys.argv) > 1 and sys.argv[1] == 'ai':                try:

        play_ai_vs_ai()

    else:                    action = int(input("输入动作编号: "))def play_game(player1, player2, game, display=True):

        play_human_vs_ai()

                    if action in valid_actions:    """

                        break    执行一局游戏

                    print("无效动作，请重新输入!")    

                except:    Args:

                    print("请输入数字!")        player1: 玩家1

        else:        player2: 玩家2

            # AI回合        game: 游戏实例

            print("AI思考中...")        display: 是否显示棋盘

            pi = mcts.get_action_prob(state, temp=0)        

            action = np.argmax(pi)    Returns:

            print(f"AI选择动作: {action}")        result: 游戏结果

            """

        state = game.get_next_state(state, action)    state = game.get_initial_state()

        current_player = game.get_player(state)

    # 游戏结束    

    game.display(state)    player1.reset()

        player2.reset()

    result = game.get_game_result(state, human_player)    

    if result == 1:    move_count = 0

        print("🎉 你赢了!")    

    elif result == -1:    if display:

        print("😢 AI赢了!")        print("\n" + "="*60)

    else:        print("Game Start!")

        print("🤝 平局!")        print("="*60)

        game.display(state)

    

def play_ai_vs_ai():    while True:

    """AI vs AI 演示"""        move_count += 1

            

    game = DotsAndBoxesGame(5, 5)        if display:

    nnet = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)            print(f"\n--- Move {move_count} ---")

                print(f"Current Player: {current_player}")

    checkpoint = torch.load('checkpoints/latest.pth')        

    nnet.load_state_dict(checkpoint['state_dict'])        # 选择玩家

            player = player1 if current_player == 0 else player2

    if torch.cuda.is_available():        

        nnet.cuda()        # 获取动作

            action = player.get_action(state, temp=0)

    nnet.eval()        

            if action is None:  # 人类玩家中断

    args = {            return None

        'num_simulations': 100,        

        'cpuct': 1.0,        if display:

        'dirichlet_alpha': 0.3,            print(f"Action: {action}")

        'dirichlet_epsilon': 0.0,        

        'temp_threshold': 0,        # 执行动作

    }        old_player = game.get_player(state)

            state = game.get_next_state(state, action, current_player)

    mcts = MCTS(game, nnet, args)        new_player = game.get_player(state)

            

    state = game.get_initial_state()        if display:

                game.display(state)

    print("\n" + "=" * 60)        

    print("点格棋 - AI vs AI 演示")        # 检查游戏是否结束

    print("=" * 60 + "\n")        result = game.get_game_ended(state)

            if result != 0:

    move_count = 0            if display:

    while not game.is_terminal(state):                print("\n" + "="*60)

        move_count += 1                print("Game Over!")

        print(f"\n--- 第 {move_count} 步 ---")                if result > 0:

                            print("Player 0 (First) wins!")

        pi = mcts.get_action_prob(state, temp=0)                elif result < 0:

        action = np.argmax(pi)                    print("Player 1 (Second) wins!")

                        else:

        print(f"玩家 {game.get_current_player(state)} 选择动作: {action}")                    print("Draw!")

        state = game.get_next_state(state, action)                print("="*60)

                    return result

        game.display(state)        

                # 更新当前玩家

        input("按回车继续...")        if new_player != old_player:

                current_player = new_player

    returns = state.returns()            if display and new_player == old_player:

    print(f"\n游戏结束! 玩家0: {returns[0]}, 玩家1: {returns[1]}")                print("(Same player continues)")





if __name__ == '__main__':def main():

    import sys    """

        人机对战主函数

    if len(sys.argv) > 1 and sys.argv[1] == 'ai':    """

        play_ai_vs_ai()    import argparse

    else:    import os

        play_human_vs_ai()    

    parser = argparse.ArgumentParser(description='Play against the AI')
    parser.add_argument('--checkpoint', type=str, default='latest.pth',
                       help='Model checkpoint to load')
    parser.add_argument('--human-first', action='store_true',
                       help='Human plays first')
    parser.add_argument('--no-mcts', action='store_true',
                       help='AI uses network only (no MCTS)')
    parser.add_argument('--simulations', type=int, default=None,
                       help='Number of MCTS simulations')
    args = parser.parse_args()
    
    # 覆盖配置
    if args.simulations is not None:
        config.NUM_SIMULATIONS = args.simulations
    
    print("\n" + "="*60)
    print("Dots and Boxes - Human vs AI")
    print("="*60)
    
    # 加载模型
    print("\nLoading model...")
    game = DotsAndBoxesGame(config.NUM_ROWS, config.NUM_COLS)
    nnet = DotsAndBoxesNNet(game, config)
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, args.checkpoint)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        nnet.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {args.checkpoint}")
        print(f"  Iteration: {checkpoint.get('iteration', 'unknown')}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("Using untrained model...")
    
    nnet.eval()
    
    # 创建玩家
    human = HumanPlayer(game)
    ai = AIPlayer(game, nnet, config, use_mcts=not args.no_mcts)
    
    if args.human_first:
        player1, player2 = human, ai
        print("\nYou are Player 0 (playing first)")
    else:
        player1, player2 = ai, human
        print("\nYou are Player 1 (playing second)")
    
    if not args.no_mcts:
        print(f"AI using MCTS with {config.NUM_SIMULATIONS} simulations")
    else:
        print("AI using network only (no MCTS)")
    
    # 开始游戏
    result = play_game(player1, player2, game, display=True)
    
    if result is not None:
        # 判断人类玩家的结果
        if args.human_first:
            human_result = result
        else:
            human_result = -result
        
        if human_result > 0:
            print("\n🎉 Congratulations! You won!")
        elif human_result < 0:
            print("\n😔 AI wins! Better luck next time.")
        else:
            print("\n🤝 It's a draw!")


if __name__ == "__main__":
    main()
