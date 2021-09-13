import numpy as numpy


stocks = ["apple","intel","amd","microsoft",
        "pfizer","moderna","johnson & johnson",
        "bitcoin","ethereum","dash","cardano",
        "nvidia","xbox"]

UsersSearch ={
  "0":["apple","intel","amd","microsoft"],
  "1":["pfizer","moderna","johnson & johnson"],
  "3":["bitcoin","ethereum","dash","cardano"],
  "4":["intel","nvidia","xbox"],
}


def build_co_occurence():
    matrix = numpy.zeros([len(stocks),len(stocks)])
    for _,k in UsersSearch.items():
        for i in k:
            for j in k:
                if i!=j:
                    matrix[stocks.index(i)][stocks.index(j)]=1
    return matrix

def search(items,matrix):
    search=[]
    for k in items:
        for i in range(0,len(stocks)):
            if matrix[stocks.index(k)][i] == 1:
                if stocks[i] not in search:
                    search.append(stocks[i])

    return search



if __name__ == '__main__':
    occ_matrix = build_co_occurence()
    print(search(["intel","apple"],occ_matrix))
