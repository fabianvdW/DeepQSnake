import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Flatten, Conv2D, Dense, Activation
from env import *
from hyperparameters import *


class DeepQModel(Model):

    def train_step(self, data):
        x, (actions, targets) = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            q_values = tf.gather(y_pred, actions, axis=1, batch_dims=1)
            loss = self.compiled_loss(targets, q_values, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(targets, q_values)
        return {m.name: m.result() for m in self.metrics}

    def get_qs(self, state):
        return self.predict(np.expand_dims(state, 0))[0]

    def get_action(self, state):
        return np.argmax(self.get_qs(state))

    def do_test_game(self, state_list=False):
        env = Env()
        if state_list:
            states = []
        game_over = False
        while not game_over:
            curr_state = np.copy(env.board)
            action = self.get_action(curr_state)
            _, reward, game_over = env.step(action)
            if state_list:
                states.append((curr_state, reward, game_over))
        if state_list:
            return env.score(), states
        else:
            return env.score()

    def evaluate_on_games(self, games=100):
        avg_score = 0
        for _ in range(games):
            avg_score += self.do_test_game(state_list=False)
        return avg_score / games


def compile_model():
    inputs = Input(shape=(WIDTH, HEIGHT, CHANNELS))
    x = Conv2D(12, kernel_size=3)(inputs)
    x = Activation("relu")(x)
    x = Conv2D(12, kernel_size=3, padding="same")(x)
    x = Activation("relu")(x)
    x = Flatten()(x)
    x = Dense(12)(x)
    x = Activation("relu")(x)
    x = Dense(12)(x)
    x = Activation("relu")(x)
    x = Dense(4)(x)
    model = DeepQModel(inputs, x)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001 * ALPHA), loss="mse")
    return model
