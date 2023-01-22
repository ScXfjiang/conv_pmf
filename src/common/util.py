def show_elapsed_time(start, end, label=None):
    sec = end - start
    hour = int(sec // 3600)
    sec = sec - hour * 3600
    min = int(sec // 60)
    sec = sec - min * 60
    print(
        "{} elapsed time:\t {} hours {} mins {} seconds".format(label, hour, min, sec)
    )
