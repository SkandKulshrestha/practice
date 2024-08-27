class Tangram:
    def __init__(self, shape, color, blank_color):
        self.shape = shape
        self.color = color
        self.blank_color = blank_color

    def fit(self, board, position):
        p_i, p_j = position
        revert_position = list()
        try:
            for i, j in self.shape:
                _p_i = p_i + i
                _p_j = p_j + j
                if board[_p_i][_p_j] != self.blank_color:
                    raise ValueError
                revert_position.append((_p_i, _p_j))
                board[_p_i][_p_j] = self.color
        except ValueError:
            print('Error!!!')
            print(revert_position)
            for i, j in revert_position:
                board[i][j] = self.blank_color

    def unfit(self, board, position):
        p_i, p_j = position
        for i, j in self.shape:
            board[p_i + i][p_j + j] = self.blank_color


class Calendar:
    def __init__(self):
        self.board = [
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            ['1', '2', '3', '4', '5', '6', '7'],
            ['8', '9', '10', '11', '12', '13', '14'],
            ['15', '16', '17', '18', '19', '20', '21'],
            ['22', '23', '24', '25', '26', '27', '28'],
            ['29', '30', '31']
        ]
        self.tangram_board = [
            ['W', 'W', 'W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W', 'W', 'W', 'W'],
            ['W', 'W', 'W', 'W', 'W', 'W', 'W'],
            ['W', 'W', 'W']
        ]
        self.tangram = list()
        self.tangram.append(Tangram([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], 'Y', 'W'))

    def print_board(self):
        print('-' * 15)
        for i in range(len(self.tangram_board)):
            for j in range(len(self.tangram_board[i])):
                print(self.tangram_board[i][j], end=' ')
            print()

    def solve_and_save(self):
        tan = self.tangram[0]
        self.print_board()
        tan.fit(self.tangram_board, (1, 0))
        self.print_board()
        tan.fit(self.tangram_board, (0, 0))
        self.print_board()
        tan.unfit(self.tangram_board, (1, 0))
        self.print_board()


if __name__ == '__main__':
    calendar = Calendar()
    calendar.solve_and_save()
