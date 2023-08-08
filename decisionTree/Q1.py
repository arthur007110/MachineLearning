
def get_attribute_error(attribute):
  error = 0
  for i in attribute:
    print(i)
    error += min(i.values())
  return error

data = open("datasets/balance+scale/balance-scale.data", "r")
data = data.read()
data = data.split("\n")
data = [i.split(",") for i in data]

LW = [{"L":0, "B":0, "R":0} for _ in range(5)]
LD = [{"L":0, "B":0, "R":0} for _ in range(5)]
RW = [{"L":0, "B":0, "R":0} for _ in range(5)]
RD = [{"L":0, "B":0, "R":0} for _ in range(5)]
  

for i in data:
  LW[int(i[1])-1][i[0]] += 1
  LD[int(i[2])-1][i[0]] += 1
  RW[int(i[3])-1][i[0]] += 1
  RD[int(i[4])-1][i[0]] += 1

error_lw = get_attribute_error(LW)
error_ld = get_attribute_error(LD)
error_rw = get_attribute_error(RW)
error_rd = get_attribute_error(RD)

print(error_lw, error_ld, error_rw, error_rd)

