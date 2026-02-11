# -*- coding: utf-8 -*-
"""
@File    : Gradient Direction and Magnitude.py
@Time    : 2026/2/11 12:39 Wednesday
@Author  : wangw
@Email   : wangw_heart@163.com
@Description: 
"""
import numpy as np

def calculate_gradient_info(gradient):
    """
    计算梯度向量的模量和方向信息

    参数:
        gradient (list or array): 梯度向量

    返回:
        dict: 包含以下键值的字典
            - 'magnitude': 梯度的L2范数（模量）
            - 'direction': 指向最陡峭上升方向的单位向量
            - 'descent_direction': 指向最陡峭下降方向的单位向量
    """
    # 将输入转换为numpy数组以便计算
    grad_array = np.array(gradient, dtype=float)

    # 计算梯度的L2范数（模量）
    magnitude = np.linalg.norm(grad_array)

    # 处理零向量的特殊情况
    if magnitude == 0:
        # 如果是零向量，方向向量也设为零向量
        direction = [0.0] * len(gradient)
        descent_direction = [0.0] * len(gradient)
    else:
        # 计算单位向量（方向向量）
        direction = (grad_array / magnitude).tolist()
        # 计算下降方向（负梯度方向）
        descent_direction = (-grad_array / magnitude).tolist()

    return {
        'magnitude': float(magnitude),
        'direction': direction,
        'descent_direction': descent_direction
    }

# 示例用法
if __name__ == "__main__":
    # 测试用例1：普通梯度向量
    gradient1 = [3.0, 4.0]
    result = calculate_gradient_info(gradient1)
    print(f"{result['magnitude']:.4f},{[round(d,4) for d in result['direction']]},{[round(d,4) for d in result['descent_direction']]}")
