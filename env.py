import numpy as np
from collections import deque

WIDTH, HEIGHT = 5, 5
CHANNEL_SNAKE_BODY, CHANNEL_SNAKE_HEAD, CHANNEL_APPLE = 0, 1, 2
CHANNELS = 3

UP, LEFT, RIGHT, DOWN = 0, 1, 2, 3
ACTIONS = [UP, LEFT, RIGHT, DOWN]
NUM_ACTIONS = 4
ACTION_CHANGES = [(0, -1), (-1, 0), (1, 0), (0, 1)]

REWARD_APPLE = 1
REWARD_GAME_FINISHED = 10
REWARD_STEP = -1 / (WIDTH + 1) / (HEIGHT + 1)
REWARD_LOSS = -2

MAX_STEPS_WITHOUT_APPLE = WIDTH * HEIGHT + 1


def random_action():
    return ACTIONS[np.random.randint(0, 4)]


class Env:
    def __init__(self):
        self.board = None
        self.snake = None
        self.apple = None
        self.steps_without_apple = None
        self.reset()

    def reset(self):
        self.board = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
        self.snake = deque()
        self.snake.append(self.random_free_pos())
        self.toggle(self.snake[0], CHANNEL_SNAKE_HEAD)
        while len(self.snake) < 3:
            tail_x, tail_y = self.snake[-1]
            new_tail_x, new_tail_y, failed = self.pseudo_apply(tail_x, tail_y, random_action())
            if not failed and self.is_free(new_tail_x, new_tail_y):
                self.snake.append((new_tail_x, new_tail_y))
                self.toggle(self.snake[-1], CHANNEL_SNAKE_BODY)
        self.apple = self.random_free_pos()
        self.toggle(self.apple, CHANNEL_APPLE)
        self.steps_without_apple = 0

    def score(self):
        return len(self.snake)

    def step(self, action):
        self.steps_without_apple += 1
        head_x, head_y = self.snake[0]
        head_x, head_y, game_over = self.pseudo_apply(head_x, head_y, action)
        reward = REWARD_STEP
        if self.board[head_x, head_y, CHANNEL_SNAKE_BODY] > 0:
            game_over = True
        self.toggle(self.snake[0], CHANNEL_SNAKE_HEAD)
        self.toggle(self.snake[0], CHANNEL_SNAKE_BODY)
        self.snake.appendleft((head_x, head_y))
        self.toggle(self.snake[0], CHANNEL_SNAKE_HEAD)
        if self.snake[0] == self.apple:
            self.steps_without_apple = 0
            self.toggle(self.apple, CHANNEL_APPLE)

            assert not game_over
            reward += REWARD_APPLE

            if self.score() == WIDTH * HEIGHT:
                return self.board, REWARD_GAME_FINISHED, True
            self.apple = self.random_free_pos()
            self.toggle(self.apple, CHANNEL_APPLE)
        else:
            self.toggle(self.snake.pop(), CHANNEL_SNAKE_BODY)

        if self.steps_without_apple >= MAX_STEPS_WITHOUT_APPLE:
            game_over = True
        if game_over:
            reward = REWARD_LOSS
        return self.board, reward, game_over

    def toggle(self, at, channel):
        self.board[at][channel] ^= 1

    def is_free(self, x, y):
        return np.sum(self.board[x, y]) == 0

    def random_free_pos(self):
        assert (np.sum(self.board) < WIDTH * HEIGHT)
        free_pos = np.where(np.sum(self.board, axis=2) == 0)
        index = np.random.randint(0, len(free_pos[0]))
        return free_pos[0][index], free_pos[1][index]

    def pseudo_apply(self, x, y, action):
        x_change, y_change = ACTION_CHANGES[action]
        x_new, y_new = np.clip(x + x_change, 0, WIDTH - 1), np.clip(y + y_change, 0, HEIGHT - 1)
        return x_new, y_new, x_new == x and y_new == y
