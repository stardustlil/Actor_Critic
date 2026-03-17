# Actor-Critic算法(QAC)

## 目标函数 

$$
\begin{align*}
    & J(\theta) = \bar{v_\pi} \quad or \quad \bar{r_\pi}\\
where \\
    &\bar{v_\pi} = \sum_{s\in{S}}d(s)v_\pi(s)=E[\sum_{t=0}^{\infty}\gamma^tR_{t+1}]\\
    &\bar{r_\pi} = \sum_{s\in{S}}d_\pi(s)r_\pi(s) = \lim_{n\to\infty}\frac{1}{n}E[\sum_{k=1}^{n}R_{t+k}]
\end{align*}
$$



## 优化方法

$$
\begin{align*}
    \theta_{t+1} & = \theta_t \,+ \, \alpha \nabla_\theta J(\theta_t) \\
                 & = \theta_t \,+ \, \alpha E_{S\sim\eta,A\sim\pi}[\nabla_\theta\ln{\pi(A|S,\theta_t)q_\pi(S,A)}]
\end{align*}
$$

由于该式中含有期望表达式，考虑 stochastic gradient-ascent 方法近似

$$
\begin{align*}
    \theta_{t+1} &= \theta_t\,+\,\alpha\nabla_\theta\ln{\pi(a_t|s_t,\theta_t)}q(s_t,a_t)
\end{align*}
$$

**怎么计算 $q_t$ ?**

1. Monte Carlo learning
2. Temporal-difference learning（AC算法所采用）

## 伪代码           

>## The simplest actor-critic algorithm (QAC)
>
>**Aim:** Search for an optimal policy by maximizing $J(\theta)$.
>
>At time step $t$ in each episode, do:
>
>Generate $a_t$ following $\pi(a \mid s_t, \theta_t)$, observe $r_{t+1}, s_{t+1}$, and then generate $a_{t+1}$ following $\pi(a \mid s_{t+1}, \theta_t)$.
>
>### Critic (value update):
>
>$$
>w_{t+1} = w_t + \alpha_w \left[ r_{t+1} + \gamma \, q(s_{t+1}, a_{t+1}, w_t) - q(s_t, a_t, w_t) \right] \nabla_w q(s_t, a_t, w_t)
>$$
>
>### Actor (policy update):
>
>$$
>\theta_{t+1} = \theta_t + \alpha_\theta \, \nabla_\theta \ln \pi(a_t \mid s_t, \theta_t) \, q(s_t, a_t, w_{t+1})
>$$

## 项目结构

```
actor_critic/
│
├── main.py
├── trainer.py
│
├── models/
│   ├── actor.py
│   └── critic.py
│
├── agents/
│   ├── a2c_agent.py
│   └── qac_agent.py
│
├── utils/
│   └── config.py   
│
└── env/
    └── make_env.py
```



