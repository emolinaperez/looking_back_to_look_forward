import yaml

class Utils:
    @staticmethod
    def read_yaml(file_path):
        """
        Reads a YAML file and returns its contents as a dictionary.

        :param file_path: Path to the YAML file.
        :return: Dictionary containing the YAML file's parameters.
        """
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {file_path} was not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")