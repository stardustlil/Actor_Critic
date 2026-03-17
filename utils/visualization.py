from pathlib import Path
import matplotlib.pyplot as plt


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_curves(rewards, actor_losses, critic_losses, output_path):
    output = Path(output_path)
    _ensure_parent(output)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    axes[0].plot(rewards, label='Episode Reward')
    axes[0].set_title('Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()

    axes[1].plot(actor_losses, color='tab:orange', label='Actor Loss')
    axes[1].set_title('Actor Loss')
    axes[1].set_xlabel('Update Step')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    axes[2].plot(critic_losses, color='tab:green', label='Critic Loss')
    axes[2].set_title('Critic Loss')
    axes[2].set_xlabel('Update Step')
    axes[2].set_ylabel('Loss')
    axes[2].legend()

    fig.savefig(output)
    plt.close(fig)
    return str(output)
