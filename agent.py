from env import *
from deepqmodel import *
from vis import *
import os
import random
from collections import deque

REPLAY_MEMORY_SIZE = 1_000_000
MIN_REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 32

TRAIN_EVERY_K_STEPS = 4
UPDATE_STABLE_MODEL_EVERY = 5
EVALUATE_EVERY = 50
EVALUATE_GAMES = 10

DISCOUNT = 0.995
EPSILON = 1
SCALE_EPSILON = 0.9995
MIN_EPSILON = 0.001


class Agent:
    def __init__(self):
        self.model = compile_model()

        self.stable_model = compile_model()
        self.stable_model.set_weights(self.stable_model.get_weights())
        self.episodes = 0
        self.env = Env()
        self.epsilon = EPSILON
        self.experience_replay = deque(maxlen=REPLAY_MEMORY_SIZE)

    def gen_experience(self):
        board = np.copy(self.env.board)
        action = self.get_action(self.env.board)
        new_state, reward, game_over = self.env.step(action)
        new_state = np.copy(new_state)
        score = self.env.score()
        if game_over:
            self.env.reset()
            new_state = None
        self.experience_replay.append((board, action, reward, new_state))
        return score

    def get_action(self, state, train=True):
        if train and np.random.rand() < self.epsilon:
            return random_action()
        else:
            return self.stable_model.get_action(state)

    def train_step(self, verbose):
        if len(self.experience_replay) < MIN_REPLAY_MEMORY_SIZE:
            return
        experiences = random.sample(self.experience_replay, BATCH_SIZE)

        def get_maxq(experience):
            if experience[3] is None:
                return 0
            else:
                return np.max(self.stable_model.get_qs(experience[3]))

        targets = np.array([[x[2] + DISCOUNT * get_maxq(x), x[1]] for x in experiences])
        x = np.array([x[0] for x in experiences])
        self.model.fit(x, targets, batch_size=BATCH_SIZE, shuffle=False, verbose=verbose)

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
                score = self.gen_experience()
                game_over = self.experience_replay[-1][3] is None
                steps += 1
                global_steps += 1
                if game_over or global_steps % TRAIN_EVERY_K_STEPS == 0:
                    self.train_step(verbose=int(game_over))
                avg_reward += self.experience_replay[-1][2]
            avg_reward /= steps
            avg_rewards_last_100.append((avg_reward * steps, steps))
            avg_scores_last_100.append(score)

            if len(self.experience_replay) < MIN_REPLAY_MEMORY_SIZE:
                i -= 1
                continue

            self.episodes += 1
            self.epsilon = np.clip(self.epsilon * SCALE_EPSILON, MIN_EPSILON, np.inf)
            if self.episodes % UPDATE_STABLE_MODEL_EVERY == 0:
                self.stable_model.set_weights(self.model.get_weights())
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
    agent.train(100000, log_path="log.csv", video_folder="./videos/", model_folder="./models/")
