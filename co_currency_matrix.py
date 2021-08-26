import numpy as numpy


products = ["apple","intel","amd","microsoft",
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
    matrix = numpy.zeros([len(products),len(products)])
    for _,k in UsersSearch.items():
        for i in k:
            for j in k:
                if i!=j:
                    matrix[products.index(i)][products.index(j)]=1
    return matrix

def search(items,matrix):
    search=[]
    for k in items:
        for i in range(0,len(products)):
            if matrix[products.index(k)][i] == 1:
                search.append(products[i])

    return search



if __name__ == '__main__':
    occ_matrix = build_co_occurence()
    print(search(["intel","pfizer"],occ_matrix))
