Synthetic single-table data with specific_combinations
==========================================


.. code-block:: python

    import pandas as pd

    from sdgx.data_connectors.csv_connector import CsvConnector
    from sdgx.data_models.metadata import Metadata
    from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
    from sdgx.synthesizer import Synthesizer
    from sdgx.utils import download_demo_data

    dataset_csv = download_demo_data()
    data_connector = CsvConnector(path=dataset_csv)

    # Specific the fixed column combinations.
    # It can be specified multiple combinations by different tuples. Here we only specify one.
    metadata = Metadata.from_dataframe(pd.read_csv(dataset_csv))
    combinations = {("education", "educational-num")}
    metadata.update({"specific_combinations": combinations})

    synthesizer = Synthesizer(
        model=CTGANSynthesizerModel(epochs=1),  # For quick demo
        data_connector=data_connector,
        metadata=metadata
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
