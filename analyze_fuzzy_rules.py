#!/usr/bin/env python3
"""
模糊规则分析脚本
用于详细分析训练好的模糊分类器的规则
"""

import argparse
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fuzzy_model import FuzzyClassifier

def load_model_and_rules(model_dir):
    """加载模型和规则"""
    print(f"从 {model_dir} 加载模型和规则...")
    
    # 加载模型
    fuzzy_classifier = joblib.load(os.path.join(model_dir, 'fuzzy_classifier.pkl'))
    
    # 加载配置
    config_file = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 尝试加载更新的规则文件
    updated_rules_file = os.path.join(model_dir, 'fuzzy_rules_updated.json')
    original_rules_file = os.path.join(model_dir, 'fuzzy_rules.json')
    
    if os.path.exists(updated_rules_file):
        fuzzy_classifier.load_rules_from_file(updated_rules_file)
        print("已加载测试后更新的规则统计")
    elif os.path.exists(original_rules_file):
        fuzzy_classifier.load_rules_from_file(original_rules_file)
        print("已加载原始规则统计")
    
    return fuzzy_classifier, config

def analyze_rule_coverage(fuzzy_classifier):
    """分析规则覆盖情况"""
    print("\n" + "="*60)
    print("规则覆盖分析")
    print("="*60)
    
    total_rules = len(fuzzy_classifier.fuzzy_rules_per_cluster)
    used_rules = sum(1 for count in fuzzy_classifier.rule_usage_count.values() if count > 0)
    unused_rules = total_rules - used_rules
    
    print(f"总规则数: {total_rules}")
    print(f"已使用规则数: {used_rules}")
    print(f"未使用规则数: {unused_rules}")
    print(f"规则使用率: {used_rules/total_rules*100:.2f}%")
    
    # 按标签分析
    label_stats = {}
    for rule_key, usage_count in fuzzy_classifier.rule_usage_count.items():
        label = rule_key[0]
        if label not in label_stats:
            label_stats[label] = {'total': 0, 'used': 0, 'usage_sum': 0}
        label_stats[label]['total'] += 1
        label_stats[label]['usage_sum'] += usage_count
        if usage_count > 0:
            label_stats[label]['used'] += 1
    
    print(f"\n按标签统计:")
    for label, stats in label_stats.items():
        usage_rate = stats['used'] / stats['total'] * 100
        avg_usage = stats['usage_sum'] / stats['total']
        print(f"  标签 {label}: {stats['used']}/{stats['total']} 规则被使用 ({usage_rate:.1f}%), 平均使用 {avg_usage:.1f} 次")
    
    return label_stats

def analyze_rule_efficiency(fuzzy_classifier):
    """分析规则效率"""
    print("\n" + "="*60)
    print("规则效率分析")
    print("="*60)
    
    # 计算规则效率指标
    usage_counts = list(fuzzy_classifier.rule_usage_count.values())
    if not usage_counts:
        print("没有规则使用统计数据")
        return
    
    total_usage = sum(usage_counts)
    mean_usage = np.mean(usage_counts)
    std_usage = np.std(usage_counts)
    
    print(f"总使用次数: {total_usage}")
    print(f"平均使用次数: {mean_usage:.2f}")
    print(f"使用次数标准差: {std_usage:.2f}")
    print(f"使用次数变异系数: {std_usage/mean_usage:.2f}")
    
    # 找出高效和低效规则
    sorted_rules = sorted(fuzzy_classifier.rule_usage_count.items(), 
                         key=lambda x: x[1], reverse=True)
    
    print(f"\n最常用的5个规则:")
    for i, (rule_key, count) in enumerate(sorted_rules[:5]):
        label, cluster_id = rule_key
        percentage = count / total_usage * 100
        print(f"  {i+1}. Rule_{label}_{cluster_id}: {count} 次 ({percentage:.1f}%)")
    
    print(f"\n最少用的5个规则:")
    for i, (rule_key, count) in enumerate(sorted_rules[-5:]):
        label, cluster_id = rule_key
        percentage = count / total_usage * 100 if total_usage > 0 else 0
        print(f"  {i+1}. Rule_{label}_{cluster_id}: {count} 次 ({percentage:.1f}%)")

def analyze_feature_importance(fuzzy_classifier):
    """分析特征重要性"""
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)
    
    # 统计每个特征在规则中的变异程度
    feature_stats = {}
    
    for rule_key, rules in fuzzy_classifier.fuzzy_rules_per_cluster.items():
        for rule in rules:
            for mf in rule['membership_functions']:
                feature_idx = mf['feature']
                feature_name = mf['feature_name']
                stats = mf['statistics']
                
                if feature_name not in feature_stats:
                    feature_stats[feature_name] = {
                        'std_values': [],
                        'range_values': [],
                        'rule_count': 0
                    }
                
                feature_stats[feature_name]['std_values'].append(stats['std'])
                feature_stats[feature_name]['range_values'].append(stats['max'] - stats['min'])
                feature_stats[feature_name]['rule_count'] += 1
    
    # 计算特征重要性指标
    print("特征统计:")
    for feature_name, stats in feature_stats.items():
        avg_std = np.mean(stats['std_values'])
        avg_range = np.mean(stats['range_values'])
        rule_count = stats['rule_count']
        
        print(f"  {feature_name}:")
        print(f"    规则数量: {rule_count}")
        print(f"    平均标准差: {avg_std:.4f}")
        print(f"    平均取值范围: {avg_range:.4f}")

def generate_rule_comparison_report(fuzzy_classifier, output_file):
    """生成规则对比报告"""
    print(f"\n生成规则对比报告到 {output_file}...")
    
    report = []
    report.append("=" * 80)
    report.append("模糊规则对比分析报告")
    report.append("=" * 80)
    
    # 按使用频率排序规则
    sorted_rules = sorted(fuzzy_classifier.rule_usage_count.items(), 
                         key=lambda x: x[1], reverse=True)
    
    report.append(f"\n规则使用频率排序:")
    report.append("-" * 40)
    
    for rank, (rule_key, usage_count) in enumerate(sorted_rules, 1):
        label, cluster_id = rule_key
        rule_id = f"Rule_{label}_{cluster_id}"
        
        # 获取规则详情
        rules = fuzzy_classifier.fuzzy_rules_per_cluster.get(rule_key, [])
        if rules:
            rule = rules[0]
            sample_count = rule['sample_count']
            
            report.append(f"{rank:2d}. {rule_id}")
            report.append(f"    使用次数: {usage_count}")
            report.append(f"    训练样本数: {sample_count}")
            report.append(f"    目标动作: {label}")
            report.append(f"    规则描述: {rule['rule_description']}")
            report.append("")
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def create_rule_dashboard(fuzzy_classifier, output_dir):
    """创建规则仪表板"""
    print(f"\n创建规则仪表板到 {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 规则使用分布饼图
    plt.figure(figsize=(10, 8))
    
    # 按标签统计使用次数
    label_usage = {}
    for rule_key, usage_count in fuzzy_classifier.rule_usage_count.items():
        label = rule_key[0]
        if label not in label_usage:
            label_usage[label] = 0
        label_usage[label] += usage_count
    
    if label_usage:
        labels = list(label_usage.keys())
        sizes = list(label_usage.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        plt.pie(sizes, labels=[f'Action {label}' for label in labels], 
                colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Rule Usage Distribution by Action Label')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, 'rule_usage_by_label.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 规则效率散点图
    plt.figure(figsize=(12, 8))
    
    x_data = []  # 训练样本数
    y_data = []  # 使用次数
    labels = []  # 规则标签
    
    for rule_key, usage_count in fuzzy_classifier.rule_usage_count.items():
        rules = fuzzy_classifier.fuzzy_rules_per_cluster.get(rule_key, [])
        if rules:
            sample_count = rules[0]['sample_count']
            x_data.append(sample_count)
            y_data.append(usage_count)
            labels.append(f"Rule_{rule_key[0]}_{rule_key[1]}")
    
    if x_data and y_data:
        plt.scatter(x_data, y_data, alpha=0.6, s=60)
        plt.xlabel('Training Sample Count')
        plt.ylabel('Usage Count')
        plt.title('Rule Efficiency: Training Samples vs Usage Count')
        
        # 添加趋势线
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            plt.plot(x_data, p(x_data), "r--", alpha=0.8)
        
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'rule_efficiency_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 特征重要性热图
    create_feature_importance_heatmap(fuzzy_classifier, output_dir)

def create_feature_importance_heatmap(fuzzy_classifier, output_dir):
    """创建特征重要性热图"""
    # 收集所有特征的统计信息
    feature_data = {}
    rule_labels = []
    
    for rule_key, rules in fuzzy_classifier.fuzzy_rules_per_cluster.items():
        rule_id = f"Rule_{rule_key[0]}_{rule_key[1]}"
        rule_labels.append(rule_id)
        
        for rule in rules:
            for mf in rule['membership_functions']:
                feature_name = mf['feature_name']
                if feature_name not in feature_data:
                    feature_data[feature_name] = []
                
                # 使用标准差作为特征重要性指标
                feature_data[feature_name].append(mf['statistics']['std'])
    
    if feature_data and rule_labels:
        # 创建数据矩阵
        feature_names = list(feature_data.keys())
        data_matrix = []
        
        for rule_idx, rule_key in enumerate(fuzzy_classifier.fuzzy_rules_per_cluster.keys()):
            row = []
            rules = fuzzy_classifier.fuzzy_rules_per_cluster[rule_key]
            
            for feature_name in feature_names:
                # 找到对应特征的标准差
                std_value = 0
                for rule in rules:
                    for mf in rule['membership_functions']:
                        if mf['feature_name'] == feature_name:
                            std_value = mf['statistics']['std']
                            break
                row.append(std_value)
            data_matrix.append(row)
        
        # 绘制热图
        plt.figure(figsize=(12, 8))
        data_matrix = np.array(data_matrix)
        
        im = plt.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='Standard Deviation')
        
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.yticks(range(len(rule_labels)), rule_labels)
        plt.xlabel('Features')
        plt.ylabel('Rules')
        plt.title('Feature Importance Heatmap (Standard Deviation)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze fuzzy classifier rules')
    parser.add_argument('--model_dir', type=str, default='dbscan_saved_model',
                       help='Directory containing the trained model')
    parser.add_argument('--output_dir', type=str, default='rule_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # 检查模型目录
    if not os.path.exists(args.model_dir):
        print(f"错误: 模型目录 {args.model_dir} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和规则
    fuzzy_classifier, config = load_model_and_rules(args.model_dir)
    
    print(f"模型配置:")
    print(f"  使用ICA: {config.get('use_ica', False)}")
    print(f"  特征维度: {config.get('final_features', 'Unknown')}")
    print(f"  总规则数: {len(fuzzy_classifier.fuzzy_rules_per_cluster)}")
    
    # 执行各种分析
    analyze_rule_coverage(fuzzy_classifier)
    analyze_rule_efficiency(fuzzy_classifier)
    analyze_feature_importance(fuzzy_classifier)
    
    # 生成报告和可视化
    comparison_report = os.path.join(args.output_dir, 'rule_comparison_report.txt')
    generate_rule_comparison_report(fuzzy_classifier, comparison_report)
    
    dashboard_dir = os.path.join(args.output_dir, 'dashboard')
    create_rule_dashboard(fuzzy_classifier, dashboard_dir)
    
    print(f"\n分析完成!")
    print(f"结果已保存到: {args.output_dir}")
    print(f"  - 规则对比报告: {comparison_report}")
    print(f"  - 可视化仪表板: {dashboard_dir}")

if __name__ == '__main__':
    main() 