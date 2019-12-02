
count = {}
print(count)
with open('data750/750size4_nemocollection.txt') as f:
    for line in f:
        if '[' not in line:
            curLabel = str(line.strip())
            if curLabel not in count:
                count[curLabel] = 0
            count[curLabel] = count[curLabel] + 1

print(count)

