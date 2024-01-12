import os
import subprocess
import sys
import unittest


class TestGithubActions(unittest.TestCase):
    def test_github_actions_workflow(self):
        # Run the GitHub Actions workflow as a subprocess
        subprocess.run(["./run_actions.sh"], check=True)
        # Check if coverage report file exists
        self.assertTrue(os.path.exists('coverage.xml'), 'Coverage report file does not exist.')

        # Capture the error logs
        try:
            subprocess.run(
                ["./run_actions.sh"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except subprocess.CalledProcessError as e:
            error_logs = e.stderr
        else:
            error_logs = ""

        # Check if there are any errors
        self.assertFalse(error_logs, f"Error logs found: {error_logs}")

    def tearDown(self):
        # Delete the error logs file after the test has run
        os.remove('error_logs.txt')

    def test_upload_coverage_report(self):
        # Run the GitHub Actions workflow and check the upload step
        result = subprocess.run(['./run_actions.sh'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, 'Upload coverage to Codecov failed.')
        # Delete the error logs file after the test has run
        os.remove('error_logs.txt')

