import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sdgx.models.components.sdv_ctgan.data_transformer import DataTransformer


class TestDataTransformerOutputColumnData(unittest.TestCase):
    def setUp(self):
        """
        Set up the testing environment before each test.

        This method prepares a DataTransformer instance and a temporary directory that will be used to store output files created during the test.
        """
        self.transformer = DataTransformer()
        # Generate random data for testing the output functionality
        self.column_data_list = [np.random.rand(10, 2), np.random.rand(10, 3)]
        # Create a temporary directory where output files can be safely created without affecting the local file system
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """
        Clean up the testing environment after each test.

        This removes the temporary directory created during setUp, ensuring no test artifacts are left behind.
        """
        os.rmdir(self.temp_dir)

    @patch("builtins.print")
    def test_output_non_npz(self, mock_print):
        """
        Test the behavior of _output_column_data when the output type is not 'npz'.

        This test verifies that if an unsupported output type is specified, the method triggers a specific print statement indicating the issue.
        """
        self.transformer._output_column_data(self.column_data_list, "invalid_type")
        # Ensure the print function was called exactly once with the expected message
        mock_print.assert_called_once_with("output type is not properly selected")

    @patch("numpy.savez")
    def test_output_npz(self, mock_savez):
        """
        Test the behavior of _output_column_data when the output type is 'npz'.

        This test checks that the method correctly calls numpy.savez with the expected filename and array data when asked to output an NPZ file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change the working directory to avoid creating files in the project directory
            os.chdir(temp_dir)
            # Call the method under test
            self.transformer._output_column_data(self.column_data_list, "npz")
            # Verify that numpy.savez was called correctly
            mock_savez.assert_called_once()
            filename, *arrays = mock_savez.call_args[0]
            # Check that the file name is as expected
            self.assertEqual(filename, "output.npz")
            # Make sure all arguments after the filename are numpy arrays, as expected
            self.assertTrue(all(isinstance(arg, np.ndarray) for arg in arrays))
            # Change back to the original directory to maintain test environment integrity
            os.chdir(os.path.dirname(temp_dir))


if __name__ == "__main__":
    unittest.main()
