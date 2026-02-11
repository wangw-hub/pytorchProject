# -*- coding: utf-8 -*-
"""
@File    : Quotient Rule for Derivatives.py
@Time    : 2026/2/11 11:54 Wednesday
@Author  : wangw
@Email   : wangw_heart@163.com
@Description: 
"""
import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.

    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative

    Returns:
        The derivative value f'(x)
    """
    # Your code here
    # 创建多项式对象
    g_poly = np.poly1d(g_coeffs)
    h_poly = np.poly1d(h_coeffs)

    # 计算导数多项式
    g_derivative = np.polyder(g_poly)
    h_derivative = np.polyder(h_poly)

    # 在点x处计算函数值
    g_x = g_poly(x)
    h_x = h_poly(x)
    g_prime_x = g_derivative(x)
    h_prime_x = h_derivative(x)

    # 应用商法则公式
    numerator = g_prime_x * h_x - g_x * h_prime_x
    denominator = h_x ** 2

    return numerator / denominator
