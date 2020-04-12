import math
import time as tm
import timeit

routes = []

def find_paths(node, cities,time, path, distance,distance1):

    path.append(node)

    if len(path) > 1:
        distance += cities[path[-2]][node]
        distance1 += time[path[-2]][node]

    if (len(cities) == len(path)) and (path[0] in cities[path[-1]]):
        path.append(path[0])
        distance += cities[path[-2]][path[0]]
        distance1 += time[path[-2]][path[0]]
        routes.append([distance,distance1, path])
        return

    #recursion
    for city in cities:
        if (city not in path) and (node in cities[city]):
            find_paths(city, dict(cities), dict(time), list(path), distance,distance1)


#ako param alfa povecamo npr alfa = 0.8 a beta = 0.2 onda favorizujemo resenja sa boljom distancom, tj. najveci udeo u sumi nam 
# uzima zapravo distanca
def tezinska_fja(alfa,beta,skal_duzina,skal_vreme):
    return alfa * skal_duzina + beta * skal_vreme

if __name__ == '__main__':

    start_time = tm.time()
    

    cities = {
        'A' : {'B':5,'C':5,'D':3,'E':2,'F':2,'G':1,'I':3,'H':1,'J':7,'K':2},
        'B' : {'A':5,'C':5,'D':4,'E':3,'F':6,'G':1,'I':4,'H':5,'J':21,'K':9},
        'C' : {'A':5,'B':23,'D':3,'E':3,'F':5,'G':42,'I':1,'H':6,'J':8,'K':21},
        'D' : {'A':3,'B':4,'C':3,'E':7,'F':7,'G':17,'I':10,'H':7,'J':6,'K':12},
        'E' : {'A':18,'B':3,'C':7,'D':34,'F':3,'G':2,'I':11,'H':32,'J':4,'K':7},
        'F' : {'A':2,'B':6,'C':5,'D':7,'E':3,'G':4,'I':5,'H':1,'J':2,'K':6},
        'G' : {'A':16,'B':5,'C':10,'D':16,'E':7,'F':14,'I':2,'H':2,'J':1,'K':5},
        'I' : {'A':20,'B':18,'C':4,'D':14,'E':32,'F':1,'G':2,'H':4,'J':4,'K':4},
        'H' : {'A':3,'B':21,'C':10,'D':11,'E':32,'F':6,'G':21,'I':3,'J':5,'K':3},
        'J' : {'A':15,'B':6,'C':8,'D':11,'E':6,'F':9,'G':12,'I':8,'H':3,'K':9},
        'K' : {'A':4,'B':12,'C':5,'D':7,'E':12,'F':22,'G':12,'I':15,'H':5,'J':2}
    }

    time = {
        'A' : {'B':3,'C':5,'D':4,'E':5,'F':2,'G':2,'I':2,'H':9,'J':3,'K':21},
        'B' : {'A':7,'C':2,'D':2,'E':5,'F':3,'G':1,'I':12,'H':9,'J':2,'K':16},
        'C' : {'A':5,'B':16,'D':5,'E':6,'F':4,'G':1,'I':5,'H':5,'J':12,'K':23},
        'D' : {'A':4,'B':2,'C':5,'E':17,'F':5,'G':23,'I':4,'H':21,'J':5,'K':7},
        'E' : {'A':5,'B':5,'C':6,'D':2,'F':3,'G':13,'I':3,'H':5,'J':14,'K':2},
        'F' : {'A':2,'B':3,'C':4,'D':5,'E':3,'G':32,'I':2,'H':4,'J':1,'K':8},
        'G' : {'A':1,'B':9,'C':10,'D':11,'E':7,'F':14,'I':3,'H':3,'J':7,'K':2},
        'I' : {'A':3,'B':5,'C':9,'D':11,'E':32,'F':1,'G':15,'H':2,'J':5,'K':5},
        'H' : {'A':13,'B':5,'C':1,'D':3,'E':12,'F':1,'G':2,'I':1,'J':4,'K':3},
        'J' : {'A':4,'B':2,'C':15,'D':11,'E':31,'F':22,'G':16,'I':8,'H':3,'K':7},
        'K' : {'A':7,'B':20,'C':5,'D':11,'E':21,'F':22,'G':12,'I':8,'H':3,'J':2}
    }

    
    

    find_paths('A', cities,time, [], 0, 0)
    print("\n")
    print("Velicina svih ruta je: ",len(routes))
    print("distance/time/route")
    # for i in routes:
        # print(i)
    
    sum_dist = 0
    sum_time = 0
    for i in routes:
        sum_dist += i[0]
        sum_time += i[1]
    for i in range(len(routes)):
        procenat_dist = routes[i][0]/sum_dist
        procenat_time = routes[i][1]/sum_time
        routes[i].append(procenat_dist)
        routes[i].append(procenat_time)
        routes[i].append(tezinska_fja(0.5,0.5,procenat_dist,procenat_time))
    print()
    print("#####################################")
    print("Nove rute:")
    print("duzina/vreme/rute/norm_duzina/norm_vreme/tezinska_fja")
    k = 0
    
    for i in sorted(routes,key=lambda route:route[5]):
        print(i)
        if k == 10:
            break
        k+=1

    end_time = tm.time()
    print(end_time-start_time)

    # print("Vreme izvrsavanja:",time.clock() - start_time," seconds")
    
        
    