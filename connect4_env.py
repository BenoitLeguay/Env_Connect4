import numpy as np


class Connect4Env:
    def __init__(self, size=6):
        self.size = size
        self.grid = np.zeros((size, size))
        self.player_names = ['red', 'blue']
        self.count_episode = 0.0

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.count_episode = 0.0

    def step(self, action):
        player_to_play = self.player_names[int(self.count_episode % 2)]
        action_value = action["value"]
        assert action["player_name"] == player_to_play, f"It is {player_to_play}'s turn"
        assert 0.0 <= action["value"] <= self.size, "invalid action"

        col = self.grid[:, action_value]
        col_is_full = all(col != 0.0)

        if col_is_full:
            return self.state(), -5, False

        row = np.argmax(col != 0.0) - 1
        self.grid[row, action_value] = (self.count_episode % 2) + 1

        reward, done = self.check_done((self.count_episode % 2) + 1)

        self.count_episode += 1
        return self.state(), reward, done

    def state(self):
        return self.grid.reshape(self.size**2)

    def check_done(self, player_to_play):
        mask = self.grid == player_to_play
        done = False
        reward = 0.0
        for (x, y), v in np.ndenumerate(mask):
            if not v:
                continue
            done = any((self.check_next(mask, 0, x, y, 1, 0),
                        self.check_next(mask, 0, x, y, 0, 1),
                        self.check_next(mask, 0, x, y, 1, -1),
                        self.check_next(mask, 0, x, y, 1, 1)))

            if done:
                reward = 10
                break

        return reward, done

    def check_next(self, grid_mask, n, x, y, shift_x, shift_y):
        if n == 3:
            return True
        try:
            next_x = x+shift_x
            next_y = y+shift_y
            if grid_mask[next_x, next_y]:
                return self.check_next(grid_mask, n + 1, next_x, next_y, shift_x, shift_y)
            else:
                return False
        except IndexError:
            return False

