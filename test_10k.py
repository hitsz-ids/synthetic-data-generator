from redirector import WriteableRedirector
with WriteableRedirector():
    import time
    from mycode.test_20_tables import fetch_data_from_sqlite, Metadata
    from sdgx.data_connectors.csv_connector import CsvConnector
    from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
    from sdgx.synthesizer import Synthesizer
    # from sdgx.utils import download_demo_data
    metadata, tables = fetch_data_from_sqlite(path='./mycode/data_sqlite.db')
    metadata = Metadata(metadata)
    metadata.get_tables()
    import pandas as pd
    result_table = tables["BookLoan"]
    x_table = ["Book", "Library", "Student", "Enrollment", "Submission"]
    x_key = ['book_id', "library_id", "student_id", "student_id","student_id"]
    x_how = ['outer', 'outer', "outer", 'outer', "outer"]
    print(f"{len(result_table)=}")
    for i, x_table_name in enumerate(x_table):
        result_table = pd.merge(result_table, tables[x_table_name], on=x_key[i], how=x_how[i])
        print(f"{len(result_table)=}")
    print(len(result_table.columns))
    result_table

    # This will download demo data to ./dataset
    #dataset_csv = download_demo_data()
    result_table.to_csv("test_10k.csv")
    dataset_csv = "test_10k.csv"
    # Create data connector for csv file
    data_connector = CsvConnector(path=dataset_csv)

    # Initialize synthesizer, use CTGAN model
    synthesizer = Synthesizer(
        model=CTGANSynthesizerModel(
            epochs=1,
            batch_size=200,
            
        ),  # For quick demo
        data_connector=data_connector,
    )

    def writetime():
        st = time.time()
        with open("time.log", "a+") as f:
            f.write(str(st) + "\n")
    # Fit the model
    writetime()
    synthesizer.fit()
    writetime()
    import pickle
    pickle.dump(synthesizer, open("model10k.pkl", "w"))