import casadi as ca
import numpy as np

# 设置问题参数
N = 10  # 时间步数
t_f = 1.0  # 最终时间
h = t_f / N  # 每个时间步的长度

# 创建优化变量
x = ca.MX.sym('x', N + 1)  # 状态变量
u = ca.MX.sym('u', N)  # 控制变量

# 创建优化问题
w = []  # 决策变量列表
w += [x]
w += [u]

# 创建约束和目标
g = []  # 约束条件
J = 0  # 目标函数

# 初始条件
g.append(x[0] - 0)  # x(0) = 0

# 构造配点方程和目标函数
for k in range(N):
    # 配点方程：x_{k+1} = x_k + h*u_k
    x_next = x[k] + h * u[k]
    g.append(x[k + 1] - x_next)

    # 目标函数：最小化 u_k^2
    J += u[k] ** 2 * h

# 终点条件
g.append(x[N] - 1)  # x(t_f) = 1

# 定义优化问题
nlp = {'x': ca.vertcat(*w), 'f': J, 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# 初始猜测
w0 = np.zeros((N + 1 + N,))  # 初始解
lbg = np.zeros((N + 1,))  # 约束下界
ubg = np.zeros((N + 1,))  # 约束上界

# 求解优化问题
sol = solver(x0=w0, lbg=lbg, ubg=ubg)

# 提取解
x_sol = sol['x'].full()[:N + 1]
u_sol = sol['x'].full()[N + 1:]

# 打印结果
print("Optimal state trajectory: ", x_sol)
print("Optimal control trajectory: ", u_sol)