

import numpy as np
import pickle
import sys
import easygui as g

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0       #未下棋的状态的哈希值为0
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
                if i == -1:
                    i = 2
                self.hash_val = self.hash_val * 3 + i
        return int(self.hash_val)

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(0, BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(0, BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # check diagonals
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, i]
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, BOARD_ROWS - 1 - i]

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum = np.sum(np.abs(self.data))
        if sum == BOARD_ROWS * BOARD_COLS:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # print the board
    def print_state(self):
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')

def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(0, BOARD_ROWS):
        for j in range(0, BOARD_COLS):
            if current_state.data[i][j] == 0:
                newState = current_state.next_state(i, j, current_symbol)
                newHash = newState.hash()
                if newHash not in all_states.keys():
                    isEnd = newState.is_end()
                    all_states[newHash] = (newState, isEnd)
                    if not isEnd:
                        get_all_states_impl(newState, -current_symbol, all_states)

def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states

# all possible board configurations
all_states = get_all_states()

class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        while True:
            yield self.p1           #生成器
            yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        while True:
            player = next(alternator)       #交换下棋者
            if print_state:                 #默认不画出棋盘。。。
                current_state.print_state()
            [i, j, symbol] = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if is_end:          #一局结束
                if print_state:
                    current_state.print_state()     #默认不画出棋盘
                return current_state.winner

# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []
        self.greedy = []

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)   #该列表中每一个状态都是一个状态对象
        self.greedy.append(True)
    #给每一种状态规定一个状态值，并且存入到estimations字典数据对应的key（hash_val)值下
    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states.keys():
            (state, is_end) = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5    #对应的状态值
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    # update value estimation
    def backup(self):
        # for debug
        # print('player trajectory')
        # for state in self.states:
        #     state.print_state()

        self.states = [state.hash() for state in self.states]#现在该列表中是对应状态的哈希值
        #td_error是两个连续状态的估计值之差，使用的时序差分法更新状态值
        for i in reversed(range(len(self.states) - 1)):
            state = self.states[i]  #state是相应state的哈希值
            td_error = self.greedy[i] * (self.estimations[self.states[i + 1]] - self.estimations[state])
            self.estimations[state] += self.step_size * td_error    #更新一开始给定的状态值

    # choose an action based on the state
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash], pos))
        # to select one of the actions of equal value at random
        np.random.shuffle(values)#打乱列表顺序
        values.sort(key=lambda x: x[0], reverse=True)#以状态值降序开始排列
        action = values[0][1]#选择状态值高的那个动作
        action.append(self.symbol)
        return action
    #存取的最优状态值
    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
time = 0
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
        self.time = 0
        self.key = None
        return

    def reset(self):
        return

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol
        return

    def backup(self, _):
        return

    def act(self):
        self.time += 1
         #人类玩家每下一个子之前画出棋盘
        if self.time == 1:
            image = 'tic.png'
            g.msgbox("您将进入井字棋游戏", image)
            msg = "选择您要下棋的位置"
            title = "井字棋游戏"
            image = 'tic.png'
            choices = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
            self.key = g.buttonbox(msg, title, choices, image)
            data = self.keys.index(self.key)
            i = data // int(BOARD_COLS)
            j = data % BOARD_COLS
            return (i, j, self.symbol)
        else:
            output = sys.stdout
            outputfile = open('tic.txt', "w")
            sys.stdout = outputfile
            self.state.print_state()
            outputfile.flush()
            # sys.stdout = outputfile
            type = sys.getfilesystemencoding()  # python编码转换到系统编码输出
            f = open('tic.txt', 'r')
            msg = ("您的选择是" + str(self.key), "结果")
            title = "tic_tac_toe"
            text = f.read()
            g.codebox(msg, title, text)
            f.close()
            msg = "选择您要下棋的位置"
            title = "井字棋游戏"
            image = 'tic.png'
            choices = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
            self.key = g.buttonbox(msg, title, choices, image)
            data = self.keys.index(self.key)
            i = data // int(BOARD_COLS)
            j = data % BOARD_COLS
            return (i, j, self.symbol)

def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)#训练时不画出棋盘
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()    #更新估计值
        player2.backup()
        judger.reset()   #每进行一局就清空状态列表
    player1.save_policy()
    player2.save_policy()

def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(0, turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))

# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()
        winner = judger.play()
        output = sys.stdout
        outputfile = open('tic.txt', "w")
        sys.stdout = outputfile
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")
        outputfile.flush()
        f = open('tic.txt', 'r')
        msg = ("结果是：")
        title = "tic_tac_toe"
        text = f.read()
        g.codebox(msg, title, text)
        f.close()
        msg = "您希望重新开始小游戏吗？"
        title = "请选择"
        if g.ccbox(msg, title):
            pass
        else:
            sys.exit(0)
        type = sys.getfilesystemencoding()  # python编码转换到系统编码输出
if __name__ == '__main__':
    train(int(100000))
    compete(int(100))
    play()

