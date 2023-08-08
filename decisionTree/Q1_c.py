
def get_attribute_error(attribute):
  error = 0
  for i in attribute:
    #print(i)
    error += min(i.values())
  return error

data = open("datasets/balance+scale/balance-scale.data", "r")
data = data.read()
data = data.split("\n")
data = [i.split(",") for i in data]

LD = [{"L":0, "B":0, "R":0} for _ in range(5)]
RW = [{"L":0, "B":0, "R":0} for _ in range(5)]
RD = [{"L":0, "B":0, "R":0} for _ in range(5)]
  
values = ["1","2","3","4","5"]
new_data = [[] for _ in range(5)]

for j in values:
  for i in data:
    if i[1] == j:
      new_data[int(j)-1].append(i)

#print(len(new_data))

for index, atr in  enumerate(new_data):
  LD = [{"L":0, "B":0, "R":0} for _ in range(5)]
  RW = [{"L":0, "B":0, "R":0} for _ in range(5)]
  RD = [{"L":0, "B":0, "R":0} for _ in range(5)]
  for row in atr:
    LD[int(row[2])-1][row[0]] += 1
    RW[int(row[3])-1][row[0]] += 1
    RD[int(row[4])-1][row[0]] += 1

  error_ld = get_attribute_error(LD)
  error_rw = get_attribute_error(RW)
  error_rd = get_attribute_error(RD)

  print(f'{index+1} taxas de erro: {error_ld}, {error_rw}, {error_rd}')