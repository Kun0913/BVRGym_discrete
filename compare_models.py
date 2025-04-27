import argparse
import numpy as np
import matplotlib.pyplot as plt
from BVRGym_DiscreteTest import test_model

def compare_models(model_paths, num_episodes=50):
    """
    比较多个模型在Dog场景中的性能
    
    参数:
        model_paths: 模型路径列表
        num_episodes: 每个模型测试的回合数
    """
    model_names = [path.split('/')[-1].replace('.pth', '') for path in model_paths]
    metrics_list = []
    
    print("\n===== 开始比较模型性能 =====")
    
    # 测试每个模型
    for i, path in enumerate(model_paths):
        print(f"\n测试模型 {i+1}/{len(model_paths)}: {model_names[i]}")
        metrics = test_model(path, num_episodes=num_episodes, visualize=False)
        metrics_list.append(metrics)
    
    # 提取关键指标进行比较
    win_rates = [m['win_rate'] * 100 for m in metrics_list]
    avg_rewards = [m['average_reward'] for m in metrics_list]
    avg_ep_lengths = [m['average_episode_length'] for m in metrics_list]
    
    # 蓝方导弹命中率
    blue_aim1_hits = [m['blue_missile_hit']['aim1'] for m in metrics_list]
    blue_aim2_hits = [m['blue_missile_hit']['aim2'] for m in metrics_list]
    total_blue_hits = [a + b for a, b in zip(blue_aim1_hits, blue_aim2_hits)]
    
    # 红方导弹命中率
    red_aim1_hits = [m['red_missile_hit']['aim1r'] for m in metrics_list]
    red_aim2_hits = [m['red_missile_hit']['aim2r'] for m in metrics_list]
    total_red_hits = [a + b for a, b in zip(red_aim1_hits, red_aim2_hits)]
    
    # 绘制比较图表
    # 1. 胜率对比
    plt.figure(figsize=(12, 8))
    plt.bar(model_names, win_rates)
    plt.xlabel('模型')
    plt.ylabel('蓝方胜率 (%)')
    plt.title('不同模型胜率对比')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_win_rate_comparison.png')
    plt.close()
    
    # 2. 平均奖励对比
    plt.figure(figsize=(12, 8))
    plt.bar(model_names, avg_rewards)
    plt.xlabel('模型')
    plt.ylabel('平均奖励')
    plt.title('不同模型平均奖励对比')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_reward_comparison.png')
    plt.close()
    
    # 3. 平均回合长度对比
    plt.figure(figsize=(12, 8))
    plt.bar(model_names, avg_ep_lengths)
    plt.xlabel('模型')
    plt.ylabel('平均回合长度')
    plt.title('不同模型平均回合长度对比')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_episode_length_comparison.png')
    plt.close()
    
    # 4. 导弹命中次数对比
    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, total_blue_hits, width, label='蓝方导弹命中')
    plt.bar(x + width/2, total_red_hits, width, label='红方导弹命中')
    
    plt.xlabel('模型')
    plt.ylabel('命中次数')
    plt.title('不同模型导弹命中次数对比')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_missile_hits_comparison.png')
    plt.close()
    
    # 5. 汇总表格
    print("\n===== 模型性能比较汇总 =====")
    print(f"{'模型名称':<15} {'胜率(%)':<10} {'平均奖励':<15} {'平均回合长度':<15} {'蓝方导弹命中':<15} {'红方导弹命中':<15}")
    print("="*75)
    
    for i, name in enumerate(model_names):
        print(f"{name:<15} {win_rates[i]:<10.2f} {avg_rewards[i]:<15.2f} {avg_ep_lengths[i]:<15.2f} {total_blue_hits[i]:<15} {total_red_hits[i]:<15}")
    
    print("\n比较完成! 结果已保存为图片文件。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs='+', required=True, 
                        help="训练好的模型路径列表")
    parser.add_argument("-n", "--num_episodes", type=int, default=50, 
                        help="每个模型测试的回合数")
    args = parser.parse_args()
    
    # 运行模型比较
    compare_models(args.models, args.num_episodes) 