import unittest

class Testes(unittest.TestCase):
    def loader(self):
        data = loader()
        self.assertEqual( type(self.data), 'numpy.ndarray')

    


if __name__ == '__main__':
    unittest.main()
