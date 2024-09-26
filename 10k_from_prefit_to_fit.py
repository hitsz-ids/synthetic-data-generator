from redirector import WriteableRedirector
with WriteableRedirector():
    import time
    def writetime():
        st = time.time()
        with open("time.log", "a+") as f:
            f.write(str(st) + "\n")

    import pickle
    synthesizer = pickle.load("10k_model_prefited.pkl")
    # Fit the model
    writetime()
    synthesizer.fit(from_prefit=True)
    writetime()
    