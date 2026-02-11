# -*- coding: utf-8 -*-
"""
@File    : product Rule for Derivatives.py
@Time    : 2026/2/11 11:33 Wednesday
@Author  : wangw
@Email   : wangw_heart@163.com
@Description: 
"""
import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.

    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i

    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    # Your code here
    def multiply_polynomials(poly1, poly2):

        result = [0] * (len(poly1) + len(poly2) - 1)
        for i in range(len(poly1)):
            for j in range(len(poly2)):
                result[i + j] += poly1[i] * poly2[j]
        return result

    def differentiate_polynomial(poly):
        """对多项式求导"""
        if len(poly) <= 1:
            return [0.0]  # 常数项的导数为0
        return [i * poly[i] for i in range(1, len(poly))]

    def round_and_trim(coefficients):
        """四舍五入并移除尾随零"""
        rounded = [round(float(c), 4) for c in coefficients]  # 强制转换为浮点数
        while len(rounded) > 1 and rounded[-1] == 0.0:       # 明确比较浮点数
            rounded.pop()
        if not rounded:                                      # 处理全零情况
            return [0.0]
        return rounded

    df = differentiate_polynomial(f_coeffs)
    dg = differentiate_polynomial(g_coeffs)

    # 应用乘积法则
    term1 = multiply_polynomials(df, g_coeffs)
    term2 = multiply_polynomials(f_coeffs, dg)

    # 相加
    result_length = max(len(term1), len(term2))
    result = [0] * result_length
    for i in range(len(term1)):
        result[i] += term1[i]
    for i in range(len(term2)):
        result[i] += term2[i]

    # 四舍五入并清理
    return round_and_trim(result)

if __name__ == '__main__':
    f = [1,2]
    g = [3,4]
    print(product_rule_derivative(f,g))