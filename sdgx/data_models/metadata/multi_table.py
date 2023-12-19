from sdgx.data_models.metadata.base import Metadata


class MultiTableMetadata(Metadata):
    table_names: list[str] = []

    pass
