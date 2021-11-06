from prediction_ability import predict


def stocks():
    print("Ingresa el stock al que deseas predecir su comportamiento")
    print("Ex:\nBitcoin -> BTC-USD\nApple -> AAPL\nTesla -> TSLA")
    stock = input("Ingresa el código: ")
    predict(stock)


if __name__=='__main':


    while True:
        print("Bienvenido al agende de inversión, digital el número de la actividad que deseas realizar")
        print("(5). Habilidad de predecir stocks")
        choose = int(input("Ingresa el número"))

        if choose==5:
            stocks()
