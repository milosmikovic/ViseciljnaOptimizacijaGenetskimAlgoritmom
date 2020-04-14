import random
import numpy as np
cities = {
        'A' : {'B':5,'C':5,'D':3,'E':2,'F':2},
        'B' : {'A':5,'C':5,'D':4,'E':3,'F':6},
        'C' : {'A':5,'B':5,'D':3,'E':7,'F':5},
        'D' : {'A':3,'B':4,'C':3,'E':7,'F':7},
        'E' : {'A':2,'B':3,'C':7,'D':7,'F':3},
        'F' : {'A':2,'B':6,'C':5,'D':7,'E':3}
    }
time = {
        'A' : {'B':3,'C':5,'D':4,'E':5,'F':2},
        'B' : {'A':3,'C':2,'D':2,'E':5,'F':3},
        'C' : {'A':5,'B':2,'D':5,'E':6,'F':4},
        'D' : {'A':4,'B':2,'C':5,'E':2,'F':5},
        'E' : {'A':5,'B':5,'C':6,'D':2,'F':3},
        'F' : {'A':2,'B':3,'C':4,'D':5,'E':3}
    }


def suma_cities(elem):
  return cities[elem[0]][elem[1]]+cities[elem[1]][elem[2]]+cities[elem[2]][elem[3]]+cities[elem[3]][elem[4]]+cities[elem[4]][elem[5]]+cities[elem[5]][elem[0]]

def suma_times(elem):
  return time[elem[0]][elem[1]]+time[elem[1]][elem[2]]+time[elem[2]][elem[3]]+time[elem[3]][elem[4]]+time[elem[4]][elem[5]]+time[elem[5]][elem[0]]

def sort_key(elem):
  return elem[2]
def sort_key_time(elem):
  return elem[3]

def selection(population):
    max=0
    k = -1
    for i in range(2):
        j = random.randrange(6)
        if population[j][1] > max:
            max = population[j][1]
            k = population[j][0]
    return k,max


all_cities=['B','C','D','E','F']
population=[]

def initialization():
  #pretpostavimo da se krece iz grada A
  for i in range(6):
    lista=list(np.random.permutation(all_cities))
    lista.insert(0,'A')
    population.append([i,lista,suma_cities(lista),suma_times(lista)])
  print(population)

print("Inicijalna populacija")
initialization()

print("Sortirano po udaljenosti")
list_sort_cities=population.copy()
list_sort_cities.sort(key=sort_key,reverse=True)
print(list_sort_cities)


print("Sortirano po vremenu")
list_sort_times=population.copy()
list_sort_times.sort(key=sort_key_time,reverse=True)
print(list_sort_times)

print("Broj elemenata u kategoriji iz liste")
size_of_category=len(population)//3
print(size_of_category)

print("Pocetna")
print(population)

i=0
for elem in list_sort_cities:
  print(elem)
  if i < size_of_category:
   population[elem[0]].append([1,0])
  elif i < 2*size_of_category:
    population[elem[0]].append([2,1]) 
  else:
     population[elem[0]].append([3,0])
  i = i+1

print(population) 

i=0
for elem in list_sort_times:
  print(elem)
  if i < size_of_category:
   population[elem[0]][4][0]+=1
  elif i < 2*size_of_category:
    population[elem[0]][4][0]+=2
  else:
     population[elem[0]][4][0]+=3
  i = i+1
print(population) 

for elem in population:
  if elem[4][0]==2:
    elem.append(0.1)
  elif elem[4][0]==3:
     elem.append(0.3)
  elif elem[4][0]==4:
    if elem[4][1]==1:
       elem.append(0.5)
    else:
       elem.append(0.7)
  elif elem[4][0]==5:
    elem.append(0.8)
  else:
    elem.append(0.9)
  i+=1

print(population)


sum_all_cities=0
sum_all_times=0

for elem in population:
  sum_all_cities+=elem[2]
  sum_all_times+=elem[3]
print("Ukupne duzine puta i vremena")
print(sum_all_cities)
print(sum_all_times)


for elem in population:
  elem.append(elem[2]/(sum_all_cities-elem[2]))
  elem.append(elem[3]/(sum_all_times-elem[3]))
  elem.append(2-elem[6]-elem[7])

print(population)


