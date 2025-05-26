#!/usr/bin/env python3
"""
使用示例脚本：展示如何使用修改后的dbscan.py训练模型
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"✓ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with return code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    return True

def main():
    print("模糊分类器训练示例")
    print("本脚本展示如何使用修改后的dbscan.py进行训练")
    
    # 检查数据文件是否存在
    data_file = 'all_win_state_action_pairs - 50000.txt'
    if not os.path.exists(data_file):
        print(f"\n警告: 数据文件 {data_file} 不存在")
        print("请确保数据文件存在后再运行此脚本")
        return
    
    # 示例1: 不使用ICA训练
    print("\n" + "="*80)
    print("示例1: 不使用ICA的训练")
    print("="*80)
    
    cmd1 = "python dbscan.py --model_dir models_no_ica --min_samples_per_label 100"
    success1 = run_command(cmd1, "Training without ICA")
    
    # 示例2: 使用ICA训练
    print("\n" + "="*80)
    print("示例2: 使用ICA的训练")
    print("="*80)
    
    cmd2 = "python dbscan.py --use_ica --ica_components 6 --model_dir models_with_ica --min_samples_per_label 100"
    success2 = run_command(cmd2, "Training with ICA")
    
    # 示例3: 使用不同ICA组件数训练
    print("\n" + "="*80)
    print("示例3: 使用不同ICA组件数的训练")
    print("="*80)
    
    cmd3 = "python dbscan.py --use_ica --ica_components 4 --model_dir models_ica_4comp --min_samples_per_label 100"
    success3 = run_command(cmd3, "Training with ICA (4 components)")
    
    # 测试训练好的模型
    if success1:
        print("\n" + "="*80)
        print("测试不使用ICA的模型")
        print("="*80)
        
        cmd_test1 = "python test.py --model_dir models_no_ica --num_episodes 5"
        run_command(cmd_test1, "Testing model without ICA")
    
    if success2:
        print("\n" + "="*80)
        print("测试使用ICA的模型")
        print("="*80)
        
        cmd_test2 = "python test.py --model_dir models_with_ica --num_episodes 5"
        run_command(cmd_test2, "Testing model with ICA")
    
    print("\n" + "="*80)
    print("所有示例完成!")
    print("="*80)
    
    print("\n可用的命令行选项:")
    print("dbscan.py:")
    print("  --use_ica                    启用ICA降维")
    print("  --ica_components N           ICA组件数量 (默认: 6)")
    print("  --data_file PATH             数据文件路径")
    print("  --model_dir DIR              模型保存目录")
    print("  --min_samples_per_label N    每个标签的最小样本数 (默认: 133)")
    print("\ntest.py:")
    print("  --model_dir DIR              模型目录")
    print("  --num_episodes N             测试回合数")
    print("  --visualize                  启用可视化")

if __name__ == '__main__':
    main() 