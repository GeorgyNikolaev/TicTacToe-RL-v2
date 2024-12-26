import random
import pickle
import numpy as np
from tqdm import tqdm


# Q-Learning параметры
ALPHA = 0.1  # Скорость обучения
GAMMA = 0.9  # Коэффициент дисконтирования
EPSILON = 0.1  # Вероятность случайного действия (epsilon-greedy)

# _____________________________________________________________________


# Функции для игры

# Инициализация игрового поля
def initialize_board() -> np.array:
    return np.zeros((3, 3), dtype=int)


# Вывод игрового поля в консоль
def print_board(board: np.array):
    symbols = {0: ' ', 1: 'X', 2: 'O'}

    for i in range(3):
        print(" | ".join(symbols[cell] for cell in board[i]))
        print('-'*9)


# Действие игрока
def make_move(board: np.array, player: int, row: int, col: int) -> bool:
    if board[row, col] == 0:
        board[row, col] = player
        return True
    else:
        return False


# Проверка на выигрыш
def check_winner(board: np.array) -> int:
    for i in range(3):
        # Проверка колонок и строчек
        if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
            return 1  # Победа X
        if np.all(board[i, :] == 2) or np.all(board[:, i] == 2):
            return 2  # Победа O

    # Проверка диагоналей
    if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
        return 1  # Победа X
    if np.all(np.diag(board) == 2) or np.all(np.diag(np.fliplr(board)) == 2):
        return 1  # Победа O

    # Проверка на ничью
    if not np.any(board == 0):
        return -1  # Ничья

    return 0  # Игра продолжается


# Представление игрового поля в строку
def get_state(board: np.array) -> str:
    return str(board.reshape(9))


# Доступные ходы
def get_available_actions(board: np.array) -> list:
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

# _____________________________________________________________________


# Q-Learning функции

# Выбор действия на основе epsilon-greedy стратегии
def choose_action(board, state, q_table, epsilon):
    available_actions = get_available_actions(board)
    if random.uniform(0, 1) < epsilon:
        return random.choice(available_actions)
    else:
        if state in q_table:
            return max(q_table[state], key=q_table[state].get, default=random.choice(available_actions))
        else:
            return random.choice(available_actions)


# Обновляет Q-таблицу на основе уравнения Q-Learning
def update_q_table(q_table: dict, state: str, action, reward: int, next_state: str):
    # Проверка на наличие state и action в таблице
    if state not in q_table:
        q_table[state] = {}
    if action not in q_table[state]:
        q_table[state][action] = 0

    # Находим максимальный Q для следующего состояния
    max_w_next = max(q_table[next_state].values(), default=0) if next_state in q_table else 0

    # Q-Learning обновление
    q_table[state][action] += ALPHA * (reward + GAMMA * max_w_next - q_table[state][action])


# Тренировка бота
def train_bot(episodes: int) -> dict:
    q_table = {}

    for _ in tqdm(range(episodes)):
        board = initialize_board()
        state = get_state(board)
        current_player = 1  # 1 = X, 2 = O

        while True:
            # Выбираем ход
            action = choose_action(board, state, q_table, EPSILON)

            # Делаем ход
            make_move(board, current_player, action[0], action[1])
            next_state = get_state(board)

            # Проверяем результат
            winner = check_winner(board)
            if winner == current_player:
                reward = 1
            elif winner == 3 - current_player:
                reward = -1
            elif winner == -1:
                reward = 0
            else:
                reward = 0

            # Обновляем q-table
            update_q_table(q_table, state, action, reward, next_state)

            if winner != 0:
                break

            # Меняем игрока
            state = next_state
            current_player = 3 - current_player

    return q_table

# _____________________________________________________________________


# Игра
def human_vs_bot(q_table):
    board = initialize_board()
    print("Добро пожаловать в крестики-нолики! Вы играете за X.")
    current_player = 1

    while True:
        print_board(board)
        if current_player == 1:  # Ход человека
            while True:
                try:
                    row, col = map(int, input("Введите строку и столбец (через пробел): ").split())
                    if not (0 <= row <= 2 and 0 <= col <= 2):
                        print("Введите числа в нужном диапазоне")
                        continue
                    if make_move(board, current_player, row, col):
                        break
                    else:
                        print("Эта клетка уже занята. Попробуйте снова.")
                except ValueError:
                    print("Некорректный ввод. Введите два числа от 0 до 2 через пробел.")
        else:  # Ход бота
            state = get_state(board)
            action = choose_action(board, state, q_table, 0)
            make_move(board, current_player, action[0], action[1])

        # Проверка победителя
        winner = check_winner(board)
        if winner != 0:
            print_board(board)
            if winner == 1:
                print("Поздравляем, вы победили!")
            if winner == 2:
                print("Бот победил. Удачи в следующий раз!")
            if winner == -1:
                print("Ничья.")
            break

        current_player = 3 - current_player

# _____________________________________________________________________


def main():
    # episodes = 100000
    # q_table = train_bot(episodes)
    # print("Обучение завершено! Теперь бот готов к игре.")
    #
    # with open("bot_brain.pkl", 'wb') as f:
    #     pickle.dump(q_table, f)
    #     print('Сохранил')

    with open("bot_brain.pkl", 'rb') as f:
        load_q_table = pickle.load(f)
        print('Прочитал')

    human_vs_bot(load_q_table)


if __name__ == "__main__":
    main()
