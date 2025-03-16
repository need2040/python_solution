import pandas as pd
import numpy as np

class BinomialTree:
    def __init__(self, T, n, sigma, r0):
        self.T = T  # Общее время
        self.n = n  # Количество периодов
        self.sigma = sigma  # Волатильность
        self.r0 = r0  # Начальная процентная ставка
        self.dt = T / n  # Длина одного периода
        self.u = np.exp(sigma * np.sqrt(self.dt))  # Фактор роста
        self.d = 1 / self.u  # Фактор снижения
        self.p = (np.exp(r0 * self.dt) - self.d) / (self.u - self.d)  # Вероятность роста
        self.q = 1 - self.p  # Вероятность снижения

    def binomial_rate(self):
        rate_tree = np.zeros((self.n + 1, self.n + 1))
        rate_tree[0, 0] = self.r0
        
        for i in range(1, self.n + 1):
            rate_tree[i, 0] = rate_tree[i - 1, 0] * self.d
            for j in range(1, i + 1):
                rate_tree[i, j] = rate_tree[i - 1, j - 1] * self.u
        
        return rate_tree

    def bond_rate(self):
        bond_tree = np.zeros((self.n + 1, self.n + 1))
        bond_tree[-1, :] = 1  # В конце срока облигация стоит 1 (100%)

        rate_tree = self.binomial_rate()
        
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                bond_tree[i, j] = (self.q * bond_tree[i + 1, j] + self.p * bond_tree[i + 1, j + 1]) / (1 + rate_tree[i, j])
        
        return bond_tree

    def bond_ratet(self, t):
        rate_tree = np.zeros((t + 1, t + 1))
        rate_tree[-1, :] = 1  # В конце срока облигация стоит 1 (100%)

        binomial_rate_tree = self.binomial_rate()
        
        for i in range(t - 1, -1, -1):
            for j in range(i + 1):
                rate_tree[i, j] = (self.q * rate_tree[i + 1, j] + self.p * rate_tree[i + 1, j + 1]) / (1 + binomial_rate_tree[i, j])
        
        return rate_tree

    def forward_bond(self, t):
        bond_rate_matrix = self.bond_rate()
        bond_ratet_matrix = self.bond_ratet(t)
        
        bond_rate_matrix_l = bond_rate_matrix[:t + 1, :t + 1]

        rate_tree = np.where(
            bond_ratet_matrix != 0,
            np.divide(bond_rate_matrix_l, bond_ratet_matrix),
            0
        )
        
        return rate_tree

    def futures_price(self, k):
        rate_tree = np.zeros((k + 1, k + 1))
        rate_tree[-1, :] = self.bond_rate()[k, 0:k + 1]
        
        for i in range(k - 1, -1, -1):
            for j in range(i + 1):
                rate_tree[i, j] = (self.q * rate_tree[i + 1, j] + self.p * rate_tree[i + 1, j + 1])
        
        return rate_tree

    def american_bond(self, strike_price, k):
        temp_tree = np.zeros((k + 1, k + 1))
        temp_tree[-1, :] = self.bond_rate()[k, 0:k + 1]
        
        for i in range(k - 1, -1, -1):
            for j in range(i + 1):
                temp_tree[i, j] = (self.q * temp_tree[i + 1, j] + self.p * temp_tree[i + 1, j + 1])

        rate_tree = np.zeros((k + 1, k + 1))
        for i in range(k + 1):
            rate_tree[-1, i] = np.maximum(0, temp_tree[-1, i] - strike_price)

        for i in range(k - 1, -1, -1):
            for j in range(i + 1):
                hold = (self.q * rate_tree[i + 1, j] + self.p * rate_tree[i + 1, j + 1]) / np.exp(self.r0)
                exercise = np.maximum(0, temp_tree[i, j] - strike_price)
                rate_tree[i, j] = np.maximum(hold, exercise)
        
        return rate_tree

    @staticmethod
    def format_tree(tree, scale=100, round_digits=2):
        tree_scaled = (tree * scale).round(round_digits)
        tree_df = pd.DataFrame(tree_scaled).apply(lambda x: f"{x}%" if x != 0 else "")
        return tree_df

    @staticmethod
    def rotate_tree(tree):
        return np.rot90(tree, k=1)

def main():
    # Константы
    T = 10
    n = 10
    sigma = 0.1
    r0 = 0.05

    # Ввод пользователя
    t = int(input("Введите значение t (период экспирации форварда): "))
    k = int(input("Введите значение k (период экспирации фьючерса): "))

    
    tree = BinomialTree(T, n, sigma, r0)

    while True:
        print("\nВыберите действие:")
        print("1. Построить дерево процентных ставок")
        print("2. Построить дерево цен облигаций")
        print("3. Построить дерево цен форвардов")
        print("4. Построить дерево цен фьючерсов")
        print("5. Построить дерево цен американского опциона")
        print("6. Выход")

        choice = input("Введите номер действия: ")

        if choice == "1":
            rate_tree = tree.binomial_rate()
            rate_tree_formatted = tree.format_tree(tree.rotate_tree(rate_tree))
            print("\nДерево процентных ставок:")
            print(rate_tree_formatted)

        elif choice == "2":
            bond_tree = tree.bond_rate()
            bond_tree_formatted = tree.format_tree(tree.rotate_tree(bond_tree))
            print("\nДерево цен облигаций:")
            print(bond_tree_formatted)

        elif choice == "3":
            forward_tree = tree.forward_bond(t)
            forward_tree_formatted = tree.format_tree(tree.rotate_tree(forward_tree))
            print("\nДерево цен форвардов:")
            print(forward_tree_formatted)

        elif choice == "4":
            futures_tree = tree.futures_price(k)
            futures_tree_formatted = tree.format_tree(tree.rotate_tree(futures_tree))
            print("\nДерево цен фьючерсов:")
            print(futures_tree_formatted)

        elif choice == "5":
            strike_price = float(input("Введите страйк для американского опциона (например, 0.7 или 0.8): "))
            american_tree = tree.american_bond(strike_price, k)
            american_tree_formatted = tree.format_tree(tree.rotate_tree(american_tree))
            print(f"\nДерево цен американского опциона (страйк {strike_price}):")
            print(american_tree_formatted)

        elif choice == "6":
            print("Выход из программы.")
            break

        else:
            print("Неверный выбор. Пожалуйста, выберите действие от 1 до 6.")

if __name__ == "__main__":
    main()