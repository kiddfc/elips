def fun(xxx):
    sayi = len(xxx)
    yyy = xxx
    for i in range(sayi):
        for x in xxx[i]:
            h = [y.remove(x) for y in yyy if x in y]

    return h


# read lines
f = open("data.txt", "r+")
words = []
c = int(f.readline())
words = f.read().splitlines()
a = []
for i in range(c):
    a.append([ord(i) - 97 for i in list(set(words[i]))])

xxx = [[] for Null in range(26)]
l_xxx = [None] * 26
min_path = 26
for i in range(26):
    for j in range(c):
        if i in a[j]:
            xxx[i].append(j)
    l_xxx[i] = len(xxx[i])

# b = [ord(i) for i in a[:]]
# b
max(l_xxx)
fun(xxx)