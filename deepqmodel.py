import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Flatten, Conv2D, Dense, Activation
from env import *
from hyperparameters import *

class AEMetric(tf.keras.metrics.Metric):
    def __init__(self, name="ae_batch", **kwargs):
        super(AEMetric, self).__init__(name=name, **kwargs)
        self.absolute_error = self.add_weight(name="ae", initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.absolute_error = tf.math.abs(y_true - y_pred)

    def result(self):
        return self.absolute_error

    def reset_states(self):
        pass


class DeepQModel(Model):
    def train_step(self, data):
        x, (actions, targets), sample_weight = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            q_values = tf.gather(y_pred, actions, axis=1, batch_dims=1)
            loss = self.compiled_loss(tf.expand_dims(targets, 1), tf.expand_dims(q_values, 1),
                                      regularization_losses=self.losses, sample_weight=sample_weight)
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
    y = Dense(1)(x)
    z = Dense(4)(x)
    y = tf.reshape(y, [-1])
    zmean = tf.reduce_mean(z, axis=1)
    out = tf.expand_dims(y-zmean,1)+z
    model = DeepQModel(inputs, out)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001 * ALPHA), loss="mse", metrics=[AEMetric()])
    return model
