# considering rows
row = int(input("Enter the value of rows: "))

for i in range(1, row + 1):
    a = i
    for j in range(1, i + 1):
        print(a, end=' ')
        a += row - j
    print()
