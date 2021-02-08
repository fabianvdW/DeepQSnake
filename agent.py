from env import *
from hyperparameters import *
from pexp_replay import *
from deepqmodel import *
from vis import *
import os
from collections import deque


class Agent:
    def __init__(self):
        self.model = compile_model()

        self.stable_model = compile_model()
        self.stable_model.set_weights(self.stable_model.get_weights())
        self.episodes = 0
        self.env = Env()
        self.epsilon = EPSILON
        self.experience_replay = PrioritizedExperienceReplay()

    def gen_experience(self):
        board = np.copy(self.env.board)
        action = self.get_action(self.env.board)
        new_state, reward, game_over = self.env.step(action)
        new_state = np.copy(new_state)
        score = self.env.score()
        if game_over:
            self.env.reset()
            new_state = None
        self.experience_replay.add_experience((board, action, reward, new_state))
        return score, game_over

    def update_stable_model(self):
        weights = []
        for (model_weights, stable_model_weights) in zip(self.model.get_weights(), self.stable_model.get_weights()):
            weights.append(stable_model_weights + STABLE_MODEL_TAU * (model_weights - stable_model_weights))
        self.stable_model.set_weights(weights)

    def get_action(self, state, train=True):
        if train and np.random.rand() < self.epsilon:
            return random_action()
        else:
            return self.stable_model.get_action(state)

    def train_step(self, verbose):
        experiences, sample_weights, indices = self.experience_replay.sample()
        sample_weights = sample_weights ** EXPONENT_B(self.epsilon)

        def get_next_q(experience):
            if experience[3] is None:
                return 0
            else:
                next_qs = self.stable_model.get_qs(experience[3])
                best_action = self.model.get_action(experience[3])
                return next_qs[best_action]

        x = np.array([x[0] for x in experiences])
        actions = np.array([x[1] for x in experiences], dtype=np.int32)
        targets = np.array([x[2] + DISCOUNT * get_next_q(x) for x in experiences])
        hist = self.model.fit(x, (actions, targets), sample_weight=sample_weights, batch_size=BATCH_SIZE, shuffle=False,
                              verbose=verbose)
        errors = hist.history["ae_batch"][0]
        self.experience_replay.update_priorities(indices, errors)

    def train(self, episodes, log_path="log.csv", video_folder="./videos/", model_folder="./models/"):
        if os.path.exists(log_path):
            os.remove(log_path)

        avg_rewards_last_100 = deque(maxlen=100)
        avg_scores_last_100 = deque(maxlen=100)
        global_steps = 0
        for i in range(episodes):
            game_over = False
            avg_reward = 0
            score = 0
            steps = 0
            while not game_over:
                score, game_over = self.gen_experience()
                steps += 1
                global_steps += 1
                if len(self.experience_replay) >= MIN_REPLAY_MEMORY_SIZE:
                    if global_steps % TRAIN_EVERY_K_STEPS == 0:
                        self.train_step(verbose=True)
                    if global_steps % UPDATE_STABLE_MODEL_EVERY_K_STEPS == 0:
                        self.update_stable_model()
                avg_reward += self.experience_replay[-1][2]
            avg_reward /= steps
            avg_rewards_last_100.append((avg_reward * steps, steps))
            avg_scores_last_100.append(score)

            if len(self.experience_replay) < MIN_REPLAY_MEMORY_SIZE:
                i -= 1
                continue

            self.episodes += 1
            self.epsilon = np.clip(self.epsilon * SCALE_EPSILON, MIN_EPSILON, np.inf)

            reward_last_100 = 0
            sum_steps = 0
            for (avg, _steps) in avg_rewards_last_100:
                reward_last_100 += avg
                sum_steps += _steps
            reward_last_100 /= sum_steps
            score_last_100 = sum(avg_scores_last_100) / len(avg_scores_last_100)
            print(
                "Finished Episode {}, avg reward {:.6f}, , avg_reward_last_100 : {:.6f}, score: {}, avg_score_last_100: {:.6f}, replay_size: {}, eps: {:.6f}".format(
                    self.episodes, avg_reward, reward_last_100, score, score_last_100, len(self.experience_replay),
                    self.epsilon))
            log_file = open(log_path, 'a+')
            log_file.write(str(self.episodes) + ";" + str(score) + ";" + str(avg_reward) + "\n")
            log_file.close()
            if self.episodes % EVALUATE_EVERY == 0:
                print("Evaluating Agent!...")
                score = self.stable_model.evaluate_on_games(EVALUATE_GAMES)
                print("Got average Score of {}.".format(score))
                log_file = open(log_path, 'a+')
                log_file.write(str(self.episodes) + "e;" + str(score) + "\n")
                log_file.close()
                print("Saving a sample game video...!")
                _, state_list = self.stable_model.do_test_game(state_list=True)
                model_name = "e{}-s{:.2f}".format(self.episodes, score)
                visualize_game(state_list, os.path.join(video_folder, "{}-{}.mp4".format(date_time(), model_name)))
                print("Saving model and model weights")
                self.stable_model.save_weights(model_folder + model_name + "_weights.h5")
                self.stable_model.save(model_folder + model_name + ".h5")


if __name__ == "__main__":
    agent = Agent()
    agent.train(100000, log_path="log2.csv", video_folder="./videos2/", model_folder="./models2/")
