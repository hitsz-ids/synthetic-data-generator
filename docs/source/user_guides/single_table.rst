Synthetic single-table data
==========================================


.. code-block:: python

    from sdgx.data_connectors.csv_connector import CsvConnector
    from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
    from sdgx.synthesizer import Synthesizer
    from sdgx.utils import download_demo_data

    dataset_csv = download_demo_data()
    data_connector = CsvConnector(path=dataset_csv)
    synthesizer = Synthesizer(
        model=CTGANSynthesizerModel(epochs=1),  # For quick demo
        data_connector=data_connector,
    )
    synthesizer.fit()
    sampled_data = synthesizer.sample(1000)
    synthesizer.cleanup()  # Clean all cache


    from sdgx.metrics.column.jsd import JSD

    JSD = JSD()


    selected_columns = ["workclass"]
    isDiscrete = True
    metrics = JSD.calculate(data_connector.read(), sampled_data, selected_columns, isDiscrete)

    print("JSD metric of column %s: %g" % (selected_columns[0], metrics))
