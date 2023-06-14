import datetime
import os


def make_run_name(admin):
    tz = datetime.timezone(datetime.timedelta(hours=9))
    day_time = datetime.datetime.now(tz=tz)
    run_name = day_time.strftime(f'%m%d_%H:%M:%S_{admin}')

    return run_name
