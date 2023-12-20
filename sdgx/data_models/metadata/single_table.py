
from typing import Any, Dict, List

from sdgx.data_models.metadata.base import Metadata
from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.exceptions import MetadataInitError

class SingleTableMetadata(Metadata):
    '''SingleTableMetadata
    
    This metadata is mainly used to describe the data types of all columns in a single data table.

    For each column, there should be an instance of the Data Type object.
    '''
    metadata_version: str = 'v1'
    primary_key: str | list[str]
    column_list: list[str]

    @classmethod
    def from_dataloader(
        cls,
        primary_key:  str,
        dataloader: DataLoader,
        max_chunk: int = 10,
        include_inspectors: list[str] | None = None,
        exclude_inspectors: list[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
    ):
        ''' Initialize a metadata from DataLoader and Inspectors
        
        Args:
            dataloader(DataLoader): the input DataLoader.

            primary_key(list(str) | str): the primary key of this table.

            max_chunk(int): max chunk count.

            include_inspectors(list[str]): data type inspectors that should included in this metadata (table).

            exclude_inspectors(list[str]): data type inspectors that should NOT included in this metadata (table).

            inspector_init_kwargs(dict): inspector args.
        '''
        
        # inspectors are not fully implemented now
        # currently, we leave these code here ...
        inspectors = InspectorManager().init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        for i, chunk in enumerate(dataloader.iter()):
            for inspector in inspectors:
                inspector.fit(chunk)
            if all(i.ready for i in inspectors) or i > max_chunk:
                break

        # initialize a SingleTableMetadata object 
        metadata = SingleTableMetadata(
            primary_key= primary_key,
            column_list= dataloader.columns()
            )
       
        
        
        return metadata
    
    def to_json(self) -> str:

        pass

    def load_from_json(self, json_str):

        pass


    pass
