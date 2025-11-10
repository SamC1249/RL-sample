"""
Policy Iteration Algorithm for Simple Grid World
"""
import numpy as np
from simple_env import SimpleGridEnv


class PolicyIteration:
    """
    Policy Iteration algorithm for solving MDPs.

    Uses iterative policy evaluation and policy improvement to find
    the optimal policy for the grid world environment.
    """

    def __init__(self, env, gamma=0.99, theta=1e-6):
        """
        Initialize Policy Iteration solver.

        Args:
            env: SimpleGridEnv instance
            gamma: Discount factor (default: 0.99)
            theta: Convergence threshold for policy evaluation (default: 1e-6)
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # Initialize random policy (uniform over all actions)
        self.policy = np.random.randint(0, env.n_actions, size=env.n_states)

        # Initialize value function
        self.V = np.zeros(env.n_states)

    def policy_evaluation(self):
        """
        Evaluate current policy using iterative policy evaluation.
        Handles stochastic transitions from gravitational dynamics.

        Updates self.V to be the value function for current policy.

        Returns:
            Number of iterations until convergence
        """
        iterations = 0

        while True:
            delta = 0
            iterations += 1

            # For each state
            for state in range(self.env.n_states):
                # Skip terminal states
                if self.env.is_terminal(state):
                    continue

                v = self.V[state]

                # Get action from current policy
                action = self.policy[state]

                # Calculate new value using stochastic transitions
                new_v = 0.0
                possible_states = self.env.get_possible_next_states(state, action)

                for next_state, prob in possible_states:
                    reward = self.env.get_reward(next_state)
                    new_v += prob * (reward + self.gamma * self.V[next_state])

                self.V[state] = new_v
                delta = max(delta, abs(v - new_v))

            # Check convergence
            if delta < self.theta:
                break

        return iterations

    def policy_improvement(self):
        """
        Improve policy by making it greedy with respect to current value function.
        Handles stochastic transitions from gravitational dynamics.

        Returns:
            Boolean indicating if policy is stable (no changes made)
        """
        policy_stable = True

        # For each state
        for state in range(self.env.n_states):
            # Skip terminal states
            if self.env.is_terminal(state):
                continue

            old_action = self.policy[state]

            # Find best action using stochastic transitions
            action_values = np.zeros(self.env.n_actions)

            for action in range(self.env.n_actions):
                # Calculate expected value for this action
                possible_states = self.env.get_possible_next_states(state, action)

                for next_state, prob in possible_states:
                    reward = self.env.get_reward(next_state)
                    action_values[action] += prob * (reward + self.gamma * self.V[next_state])

            # Update policy to best action
            best_action = np.argmax(action_values)
            self.policy[state] = best_action

            # Check if policy changed
            if old_action != best_action:
                policy_stable = False

        return policy_stable

    def solve(self, max_iterations=100, verbose=True):
        """
        Run policy iteration to find optimal policy.

        Args:
            max_iterations: Maximum number of policy iterations
            verbose: Print progress information

        Returns:
            Dictionary with results including:
                - policy: Optimal policy
                - value_function: Optimal value function
                - iterations: Number of iterations
                - converged: Whether algorithm converged
        """
        if verbose:
            print("Starting Policy Iteration...")
            print(f"States: {self.env.n_states}, Actions: {self.env.n_actions}")
            print(f"Gamma: {self.gamma}, Theta: {self.theta}")

        for iteration in range(max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Policy Evaluation
            eval_iterations = self.policy_evaluation()
            if verbose:
                print(f"Policy Evaluation converged in {eval_iterations} iterations")
                print(f"Mean value: {self.V.mean():.4f}, Max value: {self.V.max():.4f}, Min value: {self.V.min():.4f}")

            # Policy Improvement
            policy_stable = self.policy_improvement()

            if verbose:
                print(f"Policy stable: {policy_stable}")

            # Check convergence
            if policy_stable:
                if verbose:
                    print(f"\nPolicy Iteration converged in {iteration + 1} iterations!")
                return {
                    'policy': self.policy,
                    'value_function': self.V,
                    'iterations': iteration + 1,
                    'converged': True
                }

        if verbose:
            print(f"\nPolicy Iteration did not converge in {max_iterations} iterations")

        return {
            'policy': self.policy,
            'value_function': self.V,
            'iterations': max_iterations,
            'converged': False
        }

    def evaluate_policy(self, num_episodes=100, max_steps=1000):
        """
        Evaluate the learned policy by running episodes.

        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with evaluation metrics
        """
        total_rewards = []
        episode_lengths = []
        successes = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0

            for step in range(max_steps):
                # Get action from policy
                action = self.policy[state]

                # Take action
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                steps += 1

                if done:
                    if info['reached_target']:
                        successes += 1
                    break

                state = next_state

            total_rewards.append(episode_reward)
            episode_lengths.append(steps)

        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'success_rate': successes / num_episodes,
            'total_episodes': num_episodes
        }

    def get_action_name(self, action):
        """Convert action number to name."""
        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        return action_names.get(action, 'UNKNOWN')

    def visualize_policy(self, show_range=(0, 30)):
        """
        Visualize the learned policy on a subset of the grid.

        Args:
            show_range: Tuple (start, end) for grid subset to display
        """
        start, end = show_range
        grid = np.empty((end - start, end - start), dtype=str)

        action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}

        for i in range(start, end):
            for j in range(start, end):
                state = self.env._pos_to_state((i, j))

                # Mark special positions
                if (i, j) == self.env.black_hole_pos:
                    grid[i - start, j - start] = 'B'
                elif (i, j) == self.env.target_pos:
                    grid[i - start, j - start] = 'T'
                elif (i, j) in self.env.planet_positions:
                    grid[i - start, j - start] = 'P'
                else:
                    action = self.policy[state]
                    grid[i - start, j - start] = action_symbols.get(action, '?')

        print("\nPolicy Visualization (subset):")
        print("B=Black Hole, P=Planet, T=Target, Arrows=Policy Direction")
        for row in grid:
            print(' '.join(row))
