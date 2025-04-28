# 将目录下所有win state action pairs 文件合并成一个txt文件
import os

# 获取当前目录
current_dir = os.getcwd()

# 获取所有win_state_action_pairs开头的文件
win_files = [f for f in os.listdir(current_dir) if f.startswith('win_state_action_pairs') and f.endswith('.txt')]

# 输出文件名
output_file = 'all_win_state_action_pairs.txt'

# 合并所有文件内容
with open(output_file, 'w') as outfile:
    for filename in win_files:
        file_path = os.path.join(current_dir, filename)
        with open(file_path, 'r') as infile:
            outfile.write(infile.read())
            ## 确保每个文件之间有换行
            # outfile.write('\n')

print(f"已将{len(win_files)}个文件合并到{output_file}")

