import gymnasium as gym

def make_env(env_name, seed=None):
    env = gym.make(env_name)
    # Gymnasium 推荐在 reset 时设置种子，而不是使用 env.seed()
    # 因此这里不设置种子，种子由 trainer 在第一次 reset 时传入
    return env