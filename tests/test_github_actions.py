import os
import subprocess
import unittest


class TestGithubActions(unittest.TestCase):
    def test_github_actions_workflow(self):
        # Run the GitHub Actions workflow as a subprocess
        subprocess.run(["./run_actions.sh"], check=True)

        # Open the error logs file
        with open('error_logs.txt', 'r') as file:
            error_logs = file.read()

        # Check if there are any errors
        self.assertEqual(error_logs, "")

    def tearDown(self):
        # Delete the error logs file after the test has run
        os.remove('error_logs.txt')
