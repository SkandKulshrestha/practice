# considering number
n = int(input("Enter the value of n: "))

row = 1
while (row * (row + 1) // 2) < n:
    row += 1

for i in range(1, row + 1):
    a = i
    for j in range(1, i + 1):
        if a > n:
            break
        print(a, end=' ')
        a += row - j
    print()