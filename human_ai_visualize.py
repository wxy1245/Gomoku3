# from __future__ import print_function

import tkinter as tk
import pickle
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

# 棋盘颜色
# 浅蓝和青色搭配: #90FFFF, #E0FFFF
# 浅棕色和米黄色搭配: #D2B48C, #f5f5dc

# 棋子颜色
# 黑: #0F0F0F, 白: #F7F7F7

# class GameBoard(tk.Frame):
#     def __init__(self, parent, rows=7, columns=7, size=64, color='#D2B48C'):
#         """Create a new game board."""
#         self.parent = parent
#         self.rows = rows
#         self.columns = columns
#         self.size = size
#         self.color = color
#         self.turn = 0  # 0 for black, 1 for white
#         self.game_record = []

#         canvas_width = columns * size + 5 * size
#         canvas_height = rows * size + 5 * size

#         tk.Frame.__init__(self, parent)
#         self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
#                                 width=canvas_width, height=canvas_height, background="#d8d8bc")
#         self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5, anchor="center")

#         self.info_panel = tk.Text(root, width=15, font=("Arial", 14))
#         self.info_panel.pack(side="right", fill="both", padx=15, pady=15, anchor="center")
#         self.info_panel.insert(tk.END, "Game Record:\n")

#         self.canvas.bind("<Button-1>", self.on_tile_clicked)

#         self.parent.bind("<Configure>", self.on_window_resized)
#         self.shift_x = 0.12 * (parent.winfo_width() - self.info_panel.winfo_width())
#         self.shift_y = 0.0725 * (self.parent.winfo_height() + parent.winfo_width())
#         self.draw_board()

#         # Add a variable to store the last piece(center red point)
#         self.last_piece = None

#     def on_window_resized(self, event):
#         """Handle the window resize event."""
#         self.shift_x = 0.12 *(self.parent.winfo_width() - self.info_panel.winfo_width())
#         self.shift_y = 0.0725 * (self.parent.winfo_height() + self.parent.winfo_width())
#         self.canvas.delete("all")
#         self.draw_board()  # Redraw the board
#         self.redraw_pieces()  # Redraw the pieces

#     def redraw_pieces(self):
#         """Redraw the pieces on the board."""
#         for i, record in enumerate(self.game_record):
#             row, col = map(int, record.split(": ")[1].strip("()").split(", "))
#             self.draw_piece(row, col, i % 2)  # Redraw the piece

#     def draw_board(self):
#         """Draw the game board."""
#         for row in range(0, self.rows+2):
#             for col in range(0, self.columns+2):
#                 x1 = col * self.size
#                 y1 = row * self.size
#                 x2 = x1 + self.size
#                 y2 = y1 + self.size     
#                 self.canvas.create_rectangle(x1+self.shift_x, y1+self.shift_y, 
#                                             x2+self.shift_x, y2+self.shift_y, 
#                                             outline="black", fill=self.color, width=0)
                
#         for row in range(1, self.rows+1):
#             for col in range(1, self.columns+1):
#                 x1 = col * self.size
#                 y1 = row * self.size
#                 x2 = x1 + self.size
#                 y2 = y1 + self.size     
#                 self.canvas.create_rectangle(x1+self.shift_x, y1+self.shift_y, 
#                                             x2+self.shift_x, y2+self.shift_y, 
#                                             outline="black")
        
#         # Draw the two rectangles with chess pieces
#         piece_radius = self.size // 4
#         for i in range(2):
#             # Calculate the position for the rectangle
#             rect_x1 = (self.columns + 3) * self.size
#             rect_y1 = (i * 3 + 2) * self.size
#             rect_x2 = rect_x1 + 2 * self.size + self.parent.winfo_width() * 0.02
#             rect_y2 = rect_y1 + self.size

#             # Draw the rectangle
#             self.canvas.create_rectangle(rect_x1+self.shift_x, rect_y1+self.shift_y, 
#                                         rect_x2+self.shift_x, rect_y2+self.shift_y, 
#                                         outline="black")

#             # Calculate the position for the chess piece
#             piece_x = rect_x1 + self.size / 2
#             piece_y = rect_y1 + self.size / 2

#             # Draw the chess piece
#             color = "black" if i == 0 else "white"
#             self.canvas.create_oval(piece_x-piece_radius+self.shift_x, piece_y-piece_radius+self.shift_y, 
#                                     piece_x+piece_radius+self.shift_x, piece_y+piece_radius+self.shift_y, 
#                                     fill=color)
            
#             # Add a text box with "Black" or "White"
#             text = "Black" if i == 0 else "White"
#             self.canvas.create_text(piece_x+2*piece_radius+self.shift_x, piece_y-piece_radius/2+self.shift_y, anchor="nw", 
#                                     text=text, font=("Arial", 14))
            
#         # Draw row and column numbers
#         for i in range(0, self.rows+1):
#             self.canvas.create_text(self.shift_x+self.size*0.5, i*self.size+self.shift_y+self.size, anchor="e", 
#                                     text=str(i), font=("Arial", 14))
#         for i in range(0, self.columns+1):
#             self.canvas.create_text(i*self.size+self.shift_x+self.size, self.shift_y+self.size*0.55, anchor="s", 
#                                     text=str(i), font=("Arial", 14))

#     def on_tile_clicked(self, event):
#         """Handle a tile click event."""
#         col = round((event.x - self.shift_x) / self.size) - 1   #note: see the draw_board function, pane starts at location 1
#         row = round((event.y - self.shift_y) / self.size) - 1   
#         if 0 <= row <= self.rows and 0 <= col <= self.columns:
#             self.draw_piece(row, col, self.turn)
#             if self.turn == 0:
#                 self.game_record.append(f"Black: ({row}, {col})")
#             else:
#                 self.game_record.append(f"White: ({row}, {col})")

#             self.turn = 1 - self.turn  # Switch turn
#             self.update_info_panel()

#     def draw_piece(self, row, col, player):
#         """Draw a piece on the board."""
#         if player == 0:
#             color = '#0F0F0F'  # Black
#         else:
#             color = '#F7F7F7'  # White
#         x = col * self.size + self.shift_x + self.size
#         y = row * self.size + self.shift_y + self.size
#         self.canvas.create_oval(x-self.size*0.425, y-self.size*0.425, x+self.size*0.425, y+self.size*0.425, fill=color)

#         # Remove the last red dot
#         if self.last_piece is not None:
#             self.canvas.delete(self.last_piece)

#         # Draw a red dot in the center of the new piece
#         red_dot_radius = self.size * 0.1
#         self.last_piece = self.canvas.create_oval(x-red_dot_radius, y-red_dot_radius, 
#                                                   x+red_dot_radius, y+red_dot_radius, 
#                                                   fill='red')
        
#     def update_info_panel(self):
#         """Update the information panel."""
#         self.info_panel.insert(tk.END, self.game_record[-1] + "\n")  # Only insert the latest record

class GameBoard(tk.Frame):
    def __init__(self, parent, rows, columns, size=64, color='#D2B48C'):
        """Create a new game board."""
        self.parent = parent
        self.rows = rows
        self.columns = columns
        self.size = size
        self.color = color
        # self.turn = 0  # 0 for black, 1 for white
        self.game_record = []

        canvas_width = columns * size + 5 * size
        canvas_height = rows * size + 5 * size

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=canvas_width, height=canvas_height, background="#d8d8bc")
        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5, anchor="center")

        self.info_panel = tk.Text(root, width=15, font=("Arial", 14))
        self.info_panel.pack(side="right", fill="both", padx=15, pady=15, anchor="center")
        self.info_panel.insert(tk.END, "Game Record:\n")

        # self.canvas.bind("<Button-1>", self.on_tile_clicked)

        self.parent.bind("<Configure>", self.on_window_resized)
        self.shift_x = 0.12 * (parent.winfo_width() - self.info_panel.winfo_width())
        self.shift_y = 0.0725 * (self.parent.winfo_height() + parent.winfo_width())

        # Add black and white player
        self.black_player = None
        self.white_player = None
        self.draw_board()

        # Add a variable to store the last piece(center red point)
        self.last_piece = None

    def on_window_resized(self, event):
        """Handle the window resize event."""
        self.shift_x = 0.12 *(self.parent.winfo_width() - self.info_panel.winfo_width())
        self.shift_y = 0.0725 * (self.parent.winfo_height() + self.parent.winfo_width())
        self.canvas.delete("all")
        self.draw_board()  # Redraw the board
        self.redraw_pieces()  # Redraw the pieces

    def redraw_pieces(self):
        """Redraw the pieces on the board."""
        for i, record in enumerate(self.game_record):
            row, col = map(int, record.split(": ")[1].strip("()").split(", "))
            self.draw_piece(row, col, i % 2)  # Redraw the piece

    def draw_board(self):
        """Draw the game board."""
        for row in range(0, self.rows+2):
            for col in range(0, self.columns+2):
                x1 = col * self.size
                y1 = row * self.size
                x2 = x1 + self.size
                y2 = y1 + self.size     
                self.canvas.create_rectangle(x1+self.shift_x, y1+self.shift_y, 
                                            x2+self.shift_x, y2+self.shift_y, 
                                            outline="black", fill=self.color, width=0)
                
        for row in range(1, self.rows+1):
            for col in range(1, self.columns+1):
                x1 = col * self.size
                y1 = row * self.size
                x2 = x1 + self.size
                y2 = y1 + self.size     
                self.canvas.create_rectangle(x1+self.shift_x, y1+self.shift_y, 
                                            x2+self.shift_x, y2+self.shift_y, 
                                            outline="black")
        
        # Draw the two rectangles with chess pieces
        piece_radius = self.size // 4
        for i in range(2):
            # Calculate the position for the rectangle
            rect_x1 = (self.columns + 3) * self.size
            rect_y1 = (i * 3 + 2) * self.size
            rect_x2 = rect_x1 + 2 * self.size + self.parent.winfo_width() * 0.02
            rect_y2 = rect_y1 + self.size

            # Draw the rectangle
            self.canvas.create_rectangle(rect_x1+self.shift_x, rect_y1+self.shift_y, 
                                        rect_x2+self.shift_x, rect_y2+self.shift_y, 
                                        outline="black")

            # Calculate the position for the chess piece
            piece_x = rect_x1 + self.size / 2
            piece_y = rect_y1 + self.size / 2

            # Draw the chess piece
            color = "black" if i == 0 else "white"
            self.canvas.create_oval(piece_x-piece_radius+self.shift_x, piece_y-piece_radius+self.shift_y, 
                                    piece_x+piece_radius+self.shift_x, piece_y+piece_radius+self.shift_y, 
                                    fill=color)
            
            # Add a text box with "Black" or "White"
            # text = "Black" if i == 0 else "White"
            if i == 0:
                text = self.black_player
            else:
                text = self.white_player
            self.canvas.create_text(piece_x+2*piece_radius+self.shift_x, piece_y-piece_radius/2+self.shift_y, anchor="nw", 
                                    text=text, font=("Arial", 14))
            
        # Draw row and column numbers
        for i in range(0, self.rows+1):
            self.canvas.create_text(self.shift_x+self.size*0.5, i*self.size+self.shift_y+self.size, anchor="e", 
                                    text=str(i), font=("Arial", 14))
        for i in range(0, self.columns+1):
            self.canvas.create_text(i*self.size+self.shift_x+self.size, self.shift_y+self.size*0.55, anchor="s", 
                                    text=str(i), font=("Arial", 14))

    def on_tile_clicked(self, event):
        """Handle a tile click event."""
        col = round((event.x - self.shift_x) / self.size) - 1   #note: see the draw_board function, pane starts at location 1
        row = round((event.y - self.shift_y) / self.size) - 1   
        if 0 <= row <= self.rows and 0 <= col <= self.columns:
            self.human_player.set_last_move(self.rows - row, col, self.columns+1)
            
    def draw_piece(self, row, col, player):
        """Draw a piece on the board."""
        if player == 0:
            color = '#0F0F0F'  # Black
        else:
            color = '#F7F7F7'  # White
        x = col * self.size + self.shift_x + self.size
        y = row * self.size + self.shift_y + self.size
        self.canvas.create_oval(x-self.size*0.425, y-self.size*0.425, x+self.size*0.425, y+self.size*0.425, fill=color)

        # Remove the last red dot
        if self.last_piece is not None:
            self.canvas.delete(self.last_piece)

        # Draw a red dot in the center of the new piece
        red_dot_radius = self.size * 0.1
        self.last_piece = self.canvas.create_oval(x-red_dot_radius, y-red_dot_radius, 
                                                  x+red_dot_radius, y+red_dot_radius, 
                                                  fill='red')      

    def add_info(self, info):
        self.info_panel.insert(tk.END, info + "\n")
    
    def ai_vs_ai(self, model1, model2, start_player, board_width, board_height, n_in_row):
        n = n_in_row
        width, height = board_width, board_height

        try:
            self.start_player = start_player    #0 - player1(ai) first, 1 - player2(ai) first
            if self.start_player == 0:
                self.black_player = "ai player1"
                self.white_player = "ai player2"
            else:
                self.black_player = "ai player2"
                self.white_player = "ai player1"
            p1, p2 = [1, 2]

            best_policy1 = PolicyValueNet(width, height, model_file = model1)
            mcts_player1 = MCTSPlayer(best_policy1.policy_value_fn, c_puct=5, n_playout=800)    #c_puct在比赛时可以适当减小, 更多的利用已有知识; 训练时可以适当增大, 多探索
            mcts_player1.set_player_ind(p1)

            best_policy2 = PolicyValueNet(width, height, model_file = model2)
            mcts_player2 = MCTSPlayer(best_policy2.policy_value_fn, c_puct=5, n_playout=800)    #c_puct在比赛时可以适当减小, 更多的利用已有知识; 训练时可以适当增大, 多探索
            mcts_player1.set_player_ind(p2)

            self.board = Board(width=width, height=height, n_in_row=n)
            self.board.init_board(self.start_player)
            self.players = {p1: mcts_player1, p2: mcts_player2}
                
            self.parent.after(1000, self.ai_ai_game_step)  # Schedule the first game step
            
        except KeyboardInterrupt:
            print('\n\rquit')
    
    def ai_ai_game_step(self):
        p1, p2 = [1, 2]
        current_player = self.board.get_current_player()
        player_in_turn = self.players[current_player]
        move = player_in_turn.get_action(self.board)
        while move is None:
            self.parent.update()
            move = player_in_turn.get_action(self.board)
        if move is not None:
            if (current_player == p2 and self.start_player == 1) or current_player == p1 and self.start_player == 0: 
                self.game_record.append(f"Black: ({self.rows - move // self.board.width}, {move % self.board.width})")
            else:   
                self.game_record.append(f"White: ({self.rows - move // self.board.width}, {move % self.board.width})")
            
            self.board.do_move(move)
            if (self.start_player == 0 and current_player == p1) or (self.start_player == 1 and current_player == p2):
                self.draw_piece(self.rows - move // self.board.width, move % self.board.width, 0) #0:black
                self.add_info(f"Black:({self.rows - move // self.board.width}, {move % self.board.width})")
            else:
                self.draw_piece(self.rows - move // self.board.width, move % self.board.width, 1) #1:white
                self.add_info(f"White:({self.rows - move // self.board.width}, {move % self.board.width})")
            
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    if winner == p1:
                        self.add_info(f"Game end. Winner is AI Player1")
                    if winner == p2:
                        self.add_info(f"Game end. Winner is AI Player2")
                else:
                    self.add_info("Game end. Tie")
            else:
                self.parent.after(1000, self.ai_ai_game_step)

    def human_vs_ai(self, start_player, model_file, board_width, board_height, n_in_row):
        n = n_in_row
        width, height = board_width, board_height
        # model_file = f"./models_{width}_{height}_{n}_me/best_policy(leafDamp2500).model"

        try:
            self.start_player = start_player    #0 - player1(human) first, 1 - player2(ai) first
            if self.start_player == 0:
                self.black_player = "human"
                self.white_player = "ai"
            else:
                self.black_player = "ai"
                self.white_player = "human"
            p1, p2 = [1, 2]
            human = HumanPlayer(p1)

            self.human_player = human

            best_policy = PolicyValueNet(width, height, model_file = model_file)
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=800)    #c_puct在比赛时可以适当减小, 更多的利用已有知识; 训练时可以适当增大, 多探索
            mcts_player.set_player_ind(p2)

            self.board = Board(width=width, height=height, n_in_row=n)
            self.board.init_board(self.start_player)
            self.players = {p1: human, p2: mcts_player}
                
            self.canvas.bind("<Button-1>", self.on_tile_clicked)
            self.parent.after(1000, self.human_ai_game_step)  # Schedule the first game step
            
        except KeyboardInterrupt:
            print('\n\rquit')

    def human_ai_game_step(self):
        p1, p2 = [1, 2]
        current_player = self.board.get_current_player()
        player_in_turn = self.players[current_player]
        move = player_in_turn.get_action(self.board)
        while move is None:
            self.parent.update()
            move = player_in_turn.get_action(self.board)
        if move is not None:
            if (current_player == p2 and self.start_player == 1) or current_player == p1 and self.start_player == 0: 
                self.game_record.append(f"Black: ({self.rows - move // self.board.width}, {move % self.board.width})")
            else:   
                self.game_record.append(f"White: ({self.rows - move // self.board.width}, {move % self.board.width})")
            
            self.board.do_move(move)
            if (self.start_player == 0 and current_player == p1) or (self.start_player == 1 and current_player == p2):
                self.draw_piece(self.rows - move // self.board.width, move % self.board.width, 0) #0:black
                self.add_info(f"Black:({self.rows - move // self.board.width}, {move % self.board.width})")
            else:
                self.draw_piece(self.rows - move // self.board.width, move % self.board.width, 1) #1:white
                self.add_info(f"White:({self.rows - move // self.board.width}, {move % self.board.width})")
            
            end, winner = self.board.game_end()
            if end:
                if winner != -1:
                    if winner == p1:
                        self.add_info(f"Game end. Winner is Human")
                    if winner == p2:
                        self.add_info(f"Game end. Winner is AI")
                else:
                    self.add_info("Game end. Tie")
            else:
                self.parent.after(1000, self.human_ai_game_step)

class HumanPlayer(object):
    def __init__(self, player_ind):
        self.player_ind = player_ind
        self.last_move = None

    def get_action(self, board):
        if self.last_move is None:
            return None
        else:
            move = self.last_move
            self.last_move = None
            return move

    def set_last_move(self, row, col, width):
        self.last_move = row * width + col

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1360x1020")  # Set the window size

    board_width, board_height = 9, 9
    n_in_row = 5
    visualBoard = GameBoard(root, rows=board_height-1, columns=board_width-1)
    visualBoard.pack(side="top", fill="both", expand="true", padx=10, pady=10)

    visualBoard.human_vs_ai(start_player=1, 
                            model_file=f"./models_9_9_5_me/best_policy(leafDamp).model",
                            board_width=9, board_height=9, n_in_row=5) #0: human_first, 1:ai-first

    # visualBoard.ai_vs_ai(model1=f"./models_9_9_5_me/best_policy(leafDamp).model", 
    #                      model2=f"./models_9_9_5_me/best_policy(non-leaf).model", 
    #                      start_player=1, board_width=board_width, board_height=board_height, n_in_row=n_in_row)    #0: model1 first, 1:model2 first

    root.mainloop()
