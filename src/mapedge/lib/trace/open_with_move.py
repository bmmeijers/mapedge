import os


class OpenWithMove:
    """open file wrapper

    In case of write (mode='w') and file does already exists
    the existing file is moved aside
    (i.e. backupped + renamed with incremented version number)
    """

    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        base_name, extension = os.path.splitext(self.filename)
        counter = 1
        filename = self.filename

        if "w" in self.mode:
            while os.path.exists(filename):
                filename = f"{base_name}_~{counter}{extension}"
                counter += 1

            if filename != self.filename:
                # move aside the existing file
                print(f"OpenWithMove: Renaming {self.filename} to {filename}")
                os.rename(self.filename, filename)

        # return the file object with the given filename
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()


### tests

import unittest
import tempfile
import os


class TestOpenWithMove(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_open_with_move1(self):
        """
        Test that the OpenWithMove class creates a new file with the correct content
        """
        original_filename = os.path.join(self.test_dir.name, "test.txt")
        content = "Hello, World!"

        with OpenWithMove(original_filename, "w") as f:
            f.write(content)
        with open(original_filename, "r") as f:
            self.assertEqual(f.read(), content)

    def test_open_with_move2(self):
        """
        Test that the OpenWithMove class creates a new file with a unique name if a file with the original name already exists
        """
        original_filename = os.path.join(self.test_dir.name, "test.txt")
        content = "Hello, World!"
        with OpenWithMove(original_filename, "w") as f:
            f.write(content)

        new_content = "Hello, Python!"
        with OpenWithMove(original_filename, "w") as f:
            f.write(new_content)
        with open(original_filename, "r") as f:
            self.assertEqual(f.read(), new_content)

        # Test that the original file still exists and has the original content
        original_filename_1 = f"{os.path.splitext(original_filename)[0]}_~1{os.path.splitext(original_filename)[1]}"
        with open(original_filename_1, "r") as f:
            self.assertEqual(f.read(), content)


if __name__ == "__main__":
    unittest.main()
