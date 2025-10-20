from collections import deque

import numpy as np
import torch

from tqdm import tqdm

from src.components.enviroments.precomputed_thor_env import PrecomputedThorEnv
from src.components.enviroments.thor_env import ThorEnv


class RLTrainRunner:
    def __init__(self, env, agent, device=None, use_tqdm=True):
        self.env = env
        self.agent = agent
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)
        self.agent_config = agent.agent_config
        self.navigation_config = agent.navigation_config
        self.exploration_config = agent.exploration_config
        self.use_tqdm = use_tqdm

        self.total_episodes = self.agent_config.get("episodes", 500)
        self.scene_numbers = agent.scene_numbers
        self.log_buffer_size = 25

        self.ep_info_buffer = deque(maxlen=self.log_buffer_size)

    def robust_reset(self, scene_number, max_retries=3):
        for attempt in range(max_retries):
            try:
                obs = self.env.reset(scene_number=scene_number, random_start=True)
                return obs
            except TimeoutError:
                print(f"[WARNING] TimeoutError at reset, trying restart... ({attempt + 1}/{max_retries})")
                try:
                    self.env.close()
                except Exception:
                    pass

                self.env = PrecomputedThorEnv(render=False, rho=self.env.rho, max_actions=self.agent_config["num_steps"], map_version=self.exploration_config.get("map_version"))
        print("[ERROR] Multiple Timeouts at reset - skipping episode.")
        return None

    def robust_step(self, action, max_retries=3):
        for attempt in range(max_retries):
            try:
                obs = self.env.step(action)
                return obs
            except TimeoutError:
                print(f"[WARNING] TimeoutError at step, trying restart... ({attempt + 1}/{max_retries})")
                try:
                    self.env.close()
                except Exception:
                    pass


                self.env = PrecomputedThorEnv(render=False, rho=self.env.rho, max_actions=self.agent_config["num_steps"], map_version=self.exploration_config.get("map_version"))
                return None
        print("[ERROR] Multiple Timeouts at step - skipping episode.")
        return None

    from tqdm import tqdm

    def run(self):
        if self.use_tqdm:
            pbar = tqdm(range(self.total_episodes), desc="Training Episodes", ncols=160, leave=False, position=2, dynamic_ncols=False)
        else:
            pbar = DummyBar()

        episode_count = 0

        while episode_count < self.total_episodes:
            episode_scores = []
            episode_steps = []
            episode_rewards = []
            episode_losses = []
            episode_entropies = []
            episode_policy_losses = []
            episode_value_losses = []
            episode_stds = []

            for scene_number in self.scene_numbers:
                obs = self.robust_reset(scene_number)
                if obs is None:
                    continue
                episode_reward = 0
                episode_steps_scene = 0

                unfinished_episode = False
                while not (obs.terminated or obs.truncated):
                    action, lssg_h, gssg_h, policy_h, last_a, value = self.agent.get_action(obs)
                    next_obs = self.robust_step(action)
                    if next_obs is None:
                        unfinished_episode = True
                        break
                    reward = next_obs.reward
                    done = next_obs.terminated or next_obs.truncated
                    agent_pos = next_obs.info.get("agent_pos", None)

                    hiddens = (lssg_h, gssg_h, policy_h)
                    self.agent.rollout_buffers.add(obs.state, action, reward, done, hiddens, last_a, agent_pos)

                    episode_reward += reward
                    episode_steps_scene += 1

                    obs = next_obs

                if unfinished_episode:
                    self.agent.reset()
                    continue

                bootstrap_value = self.agent.peek_value(obs)
                self.agent.rollout_buffers.set_bootstrap_value(bootstrap_value)

                result = self.agent.update()
                loss = result.get("loss", None)
                policy_loss = result.get("policy_loss", None)
                value_loss = result.get("value_loss", None)
                entropy = result.get("entropy", None)
                std = result.get("ret_std", None)

                final_score = obs.info.get("score", 0.0)
                episode_scores.append(final_score)
                episode_steps.append(episode_steps_scene)
                episode_rewards.append(episode_reward)
                episode_losses.append(loss)
                episode_entropies.append(entropy)
                if policy_loss is not None:
                    episode_policy_losses.append(policy_loss)
                if value_loss is not None:
                    episode_value_losses.append(value_loss)
                if std is not None:
                    episode_stds.append(std)

            mean_score = np.mean(episode_scores)
            mean_steps = np.mean(episode_steps)
            mean_reward = np.mean(episode_rewards)
            mean_loss = np.mean(episode_losses)

            pbar.set_postfix(
                {"Loss": f"{mean_loss:.3f}", "Mean Score (last ep)": f"{mean_score:.2f}", "Mean Steps (last ep)": f"{mean_steps:4.1f}"}
            )
            pbar.update(1)
            pbar.refresh()  # Force re-render to prevent bar from lagging

            self.ep_info_buffer.append({"reward": mean_reward, "steps": mean_steps, "score": mean_score})

            episode_count += 1

        pbar.close()


class DummyBar:
    def update(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

    def close(self):
        pass