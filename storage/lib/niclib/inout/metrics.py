def print_metrics_dict(metrics_dict):
    metrics_names, metrics_list = [], []
    for k, v in metrics_dict.items():
        metrics_names.append(k)
        metrics_list.append(v)
    print_metrics_list(metrics_list, metrics_names)

def save_metrics_dict(filepath, metrics_dict):
    metrics_names, metrics_list = [], []
    for k, v in metrics_dict.items():
        metrics_names.append(k)
        metrics_list.append(v)
    save_metrics_list(filepath, metrics_list, metrics_names)


def print_metrics_list(metrics_list, case_names=None):
    if type(metrics_list) is not list:
        metrics_list = [metrics_list]

    if case_names is None:
        case_names = range(len(metrics_list))

    #print("\n{:^40}".format("Metrics"))
    print("{:<16}".format('Sample'), end='')
    for metric_name, metric_value in sorted(metrics_list[0].items()):
        print("  {:>8}".format(metric_name), end='')
    print("")

    for i, metrics in enumerate(metrics_list):
        print("{:<16}".format(case_names[i]), end='')
        for metric_name, metric_value in sorted(metrics.items()):
            print("  {:<08.3}".format(metric_value), end='')
        print("")
    print("")

def save_metrics_list(filepath, metrics_list, case_names=None):
    if type(metrics_list) is not list:
        metrics_list = [metrics_list]

    if case_names is None:
        case_names = range(len(metrics_list))

    with open(filepath, 'w') as f:
        f.write("{},".format('thresh_size'))
        for metric_name, metric_value in metrics_list[0].items():
            f.write("{},".format(metric_name))
        f.write('\n')

        for i, metrics in enumerate(metrics_list):
            f.write("{},".format(case_names[i]))
            for metric_name, metric_value in metrics.items():
                f.write("{},".format(metric_value))
            f.write('\n')
        f.write('\n')