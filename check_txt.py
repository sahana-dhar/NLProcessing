class TextChecker:

    class InvalidDocumentType(Exception):
        pass



    @staticmethod
    def check_txt(f):
        """
        takes in a function and checks if the first argument, the filename is a txt file
        """

        def wrapper(filename, *args, **kwargs):
            """assures that the file is a txt or it will raise an error"""

            # check if the filename was a txt file
            if filename[::-1][:4][::-1] != '.txt':
                raise TextChecker.InvalidDocumentType(f'The parser was expecting a .txt but '
                                                      f'instead received {filename}. If you would like to parse this'
                                                      f'file, you will need to use a custom parser.')

            # run the function
            val = f(filename, *args, **kwargs)
            # check the return type

            return val

        return wrapper
