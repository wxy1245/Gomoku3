import pickle
import os

board_width, board_height, n_in_row = 9, 9, 5

games_savedirpath = f"./games_saved_data/{board_width}_{board_height}_{n_in_row}/"

# 读取第一个文件
filename1 = 'data1(human_1+ai_probs).pkl'
games_data1 = []
if os.path.exists(games_savedirpath + filename1):
    with open(games_savedirpath + filename1, 'rb') as f:
        while True:
            try:
                games_data1.append(pickle.load(f))
            except EOFError:
                break

# # 只保留后面100项数据
# games_data1 = games_data1[-100:]

# 读取第二个文件
filename2 = 'data2(human_1+ai_probs).pkl'
games_data2 = []
if os.path.exists(games_savedirpath + filename2):
    with open(games_savedirpath + filename2, 'rb') as f:
        while True:
            try:
                games_data2.append(pickle.load(f))
            except EOFError:
                break

# 合并两个列表
games_data_merge = games_data1 + games_data2

# 保存合并后的数据到新的pkl文件
filename_merge = 'Merge_data(data1+data2).pkl'
with open(games_savedirpath + filename_merge, 'wb') as f:
    for data in games_data_merge:
        pickle.dump(data, f)
