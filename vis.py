from env import *
import time
import datetime
import cv2


def date_time():
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H_%M_%S")


RED = np.array([0, 45, 191])
BLUE = np.array([255, 165, 0])
GREEN = np.array([121, 228, 0])
BACKGROUND_COLOR = [np.array([96, 96, 96]), np.array([160, 160, 160])]


def visualize_game(state_list, save_path, img_size=(800, 800)):
    top_pad = 60
    padded_img_size = (img_size[0], img_size[1] + top_pad)
    acc_reward = 0

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(save_path, fourcc, 5, padded_img_size)

    for i, (state, reward, game_over) in enumerate(iter(state_list)):
        acc_reward += reward
        width, height = state.shape[0], state.shape[1]
        img = np.zeros((width, height, 3), dtype=np.uint8)
        for x in range(width):
            for y in range(height):
                if state[x, y, CHANNEL_SNAKE_BODY] > 0:
                    img[x, y] = GREEN
                elif state[x, y, CHANNEL_SNAKE_HEAD] > 0:
                    img[x, y] = BLUE
                elif state[x, y, CHANNEL_APPLE] > 0:
                    img[x, y] = RED
                else:
                    img[x, y] = BACKGROUND_COLOR[(x + y) % 2]
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_NEAREST)
        padded_img = np.zeros((img_size[0] + top_pad, img_size[1], 3), dtype=np.uint8)
        padded_img[top_pad:, :, :] = img
        padded_img[:top_pad, :, :] = 255
        score = np.sum(state[:, :, CHANNEL_SNAKE_BODY]) + 1
        cv2.putText(padded_img,
                    "Score: {}, Steps: {}, Reward: {:.4f}, Acc. Reward: {:.4f}".format(score, i, reward, acc_reward), (10, 40),
                    cv2.FONT_ITALIC, .8, color=(0, 0, 0))
        if i == len(state_list) - 1:
            if reward != REWARD_GAME_FINISHED:
                cv2.putText(padded_img, "Game Over!", (int(padded_img_size[0] / 2) - 200, int(padded_img_size[1] / 2)),
                            cv2.FONT_ITALIC,
                            3, color=(0, 45, 191), thickness=5)
            else:
                cv2.putText(padded_img, "Game Won!", (int(padded_img_size[0] / 2) - 200, int(padded_img_size[1] / 2)),
                            cv2.FONT_ITALIC,
                            3, color=(0, 45, 191), thickness=5)
        writer.write(padded_img)
    writer.release()


if __name__ == "__main__":
    env = Env()
    state_list = []
    game_over = False
    actions = [UP, LEFT, DOWN, RIGHT]
    while not game_over:
        curr_state = np.copy(env.board)
        _, reward, game_over = env.step(actions[len(state_list) % 4])
        state_list.append((curr_state, reward, game_over))
    print(len(state_list))
    visualize_game(state_list, "test.mp4")
