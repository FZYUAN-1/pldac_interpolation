

def moindre_c(X_predit, X_test):
    #print(X_predit, X_test)
    return ((X_predit-X_test)**2).sum()

def vald_crois(f,datax, n):
    N = len(datax)
    gap = int(N/n)
    err = []
    for i in range(n):
        X_test = datax[i*gap:(i+1)*gap]
        X_train = np.concatenate((datax[0:i*gap],datax[(i+1)*gap:]), axis = 0), np.concatenate((datay[0:i*gap],datay[(i+1)*gap:]), axis = 0)
        f.fit(X_train)
        err.append(1 - f.score(X_test, Y_test))
    return err

