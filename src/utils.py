import datetime
import random


def random_date(start, end):
    """
    Return a random date between two dates, extremes included
    :param start: the beginning of the date interval
    :param end: the end of the date interval, later or equal than start
    :return: a random date
    """
    assert start <= end, "Start date should be lower or equal to end date"
    days = random.randint(0, (end - start).days)
    return start + datetime.timedelta(days=days)
