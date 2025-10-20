from collections import deque
import numpy as np
import optuna
import torch
from tqdm import tqdm

from src.optim.runners.train_runner import RLTrainRunner
from src.reinforcement.a2c_agent import A2CAgent


class RLTrialRunner:
    def __init__(
        self,
        trial,
        env,
        navigation_config,
        agent_config,
        exploration_config,
        device,
        params,
        block_size=25,
        alpha=0.01,
        num_agents=None,
        max_episodes=200,
        n_jobs=1,
    ):
        self.trial = trial
        self.env = env
        self.navigation_config = navigation_config
        self.device = device
        self.block_size = block_size
        self.alpha = alpha
        self.params = params
        self.max_episode = max_episodes
        self.max_blocks = int(self.max_episode / self.block_size)
        agent_config["num_steps"] = self.block_size
        self.agent_config = agent_config
        self.num_agents = num_agents if num_agents is not None else 1
        self.n_jobs = n_jobs
        self.use_tqdm = n_jobs == 1  # Use tqdm only if running in a single process
        self.multiagent = self.num_agents > 1
        self.agents = []
        for _ in range(self.num_agents):
            if agent_config["name"] == "a2c":
                agent = A2CAgent(env=env, navigation_config=navigation_config, agent_config=agent_config, exploration_config=exploration_config)
            else:
                raise Exception("Unknown agent")
            train_runner = RLTrainRunner(
                env=self.env, agent=agent, device=torch.device("cpu") if self.multiagent else self.device, use_tqdm=self.use_tqdm
            )
            self.agents.append((agent, train_runner))

        self.recent_scores = deque(maxlen=200)  # keep last 200 episode scores across the whole trial
        self.recent_steps = deque(maxlen=200)  # keep last 200 episode steps across the whole trial

    def run(self):
        all_scores, all_steps = [], []

        # decide whether to use tqdm or simple prints
        use_tqdm = self.n_jobs == 1

        params_str = ", ".join(f"{k}={round(v, 5)}" for k, v in self.params.items())

        # Outer loop: blocks
        if use_tqdm:
            outer_iter = tqdm(
                range(self.max_blocks), desc=f"Trial {self.trial.number} Blocks ({params_str})", ncols=160, leave=False, position=0
            )

        else:
            print(f"[Trial {self.trial.number}] Starting {self.max_blocks} blocks ({params_str})")
            outer_iter = range(self.max_blocks)

        for block_idx in outer_iter:
            block_scores, block_steps = [], []

            # Inner loop: agents per block
            if use_tqdm:
                agents_iter = tqdm(range(self.num_agents), desc="Agents", ncols=160, leave=False, position=1)
            else:
                agents_iter = range(self.num_agents)

            for i, (agent, runner) in enumerate(self.agents):
                if use_tqdm:
                    agents_iter.set_description(f"Agent {i + 1}")

                if self.multiagent:
                    # Move agent to GPU if multiple agents
                    agent.to(self.device)
                    runner.device = self.device

                runner.ep_info_buffer.clear()
                runner.total_episodes = self.block_size
                runner.run()

                scores = [ep["score"] for ep in runner.ep_info_buffer]
                steps = [ep["steps"] for ep in runner.ep_info_buffer]
                block_scores.extend(scores)
                block_steps.extend(steps)

                if self.multiagent:
                    # Move back to CPU to free GPU memory
                    agent.to("cpu")
                    runner.device = torch.device("cpu")

                if use_tqdm:
                    agents_iter.update(1)
                    agents_iter.set_postfix(
                        {
                            "objective": f"{self.compute_objective(np.mean(block_scores), np.mean(block_steps)):.3f}",
                            "steps": f"{np.mean(block_steps):.0f}",
                            "score": f"{np.mean(block_scores):.3f}",
                        }
                    )

            # compute block means
            mean_score = np.mean(block_scores)
            mean_steps = np.mean(block_steps)
            all_scores.extend(block_scores)
            all_steps.extend(block_steps)
            self.recent_scores.extend(block_scores)
            self.recent_steps.extend(block_steps)

            # Report to Optuna and possibly prune
            if self.trial is not None:
                obj = self.compute_objective(mean_score, mean_steps)
                self.trial.report(obj, block_idx)
                if self.trial.should_prune():
                    # write to console even ohne tqdm
                    print(f"[Trial {self.trial.number}] Pruned at block {block_idx}")
                    raise optuna.exceptions.TrialPruned()

            if use_tqdm:
                outer_iter.set_postfix({"objective": f"{obj:.3f}", "steps": f"{mean_steps:4.1f}", "score": f"{mean_score:.3f}"})
            else:
                # simple print per block when parallel jobs
                print(
                    f"[Trial {self.trial.number}] "
                    f"Block {block_idx+1}/{self.max_blocks} – "
                    f"obj={obj:.3f}, steps={mean_steps:4.1f}, score={mean_score:.3f}"
                )

        if use_tqdm:
            outer_iter.close()

        # final summary
        recent_scores_list = list(self.recent_scores) if len(self.recent_scores) > 0 else all_scores
        recent_steps_list = list(self.recent_steps) if len(self.recent_steps) > 0 else all_steps

        final_obj = self.compute_objective(np.mean(recent_scores_list), np.mean(recent_steps_list))
        print(f"[Trial {self.trial.number}] Completed – final obj={final_obj:.3f}, steps={np.mean(recent_steps_list):.0f}")

        return final_obj

    @staticmethod
    def compute_objective(mean_score, mean_steps):
        if mean_steps <= 25:
            return mean_score
        elif mean_steps <= 30:
            # Linear penalty from 1.0 to 0.9 over 5 steps
            penalty = 1.0 - ((mean_steps - 25) / 5) * 0.1
        else:
            penalty = 0.0

        return mean_score * max(penalty, 0.0)