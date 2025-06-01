import os

def make_folder_if_not_exists(path):
    """
    Creates a folder (and parents) if it doesn't exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def date_to_str(date_obj):
    """
    Converts a datetime.date or Timestamp to 'YYYY-MM-DD' string.
    """
    return date_obj.strftime("%Y-%m-%d")
