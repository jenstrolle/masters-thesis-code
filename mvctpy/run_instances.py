if __name__ == "__main__":
    import argparse
    import time
    from .alns import ALNS
    from .tsp_to_mvctp import tsp_to_mvctp_instance


    desc = "Run ALNS method on a set of instances."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--partial", help="run on a subset of instances", action="store_true")
    parser.add_argument("--runs", help="number of runs", default=1)

    args = parser.parse_args()

    # default parameters
    params = {
        'delta': 2,
        'lambd': 3,
        'eta': 2,
        'alpha': 0.90,
        'r': 0.11,
        's1': 31,
        's2': 20,
        's3': 3,
        'epsilon': 0.36,
        'w': 0.09,
        'c': 0.99975,
        'tau': 85,
        'phi':25000,
        'verbose': True,
        'plot':True
    }

    # run on some instances

    if args.partial:
        instances = [
            'A-1-100-100-4.tsp'
        ]

    else:
        # assumes a file called instance_names exists in the instances folder
        with open('../instances/instance_names', 'r') as f:
            instances = f.read().split('\n')[:-1]

        # disable plots and verbose output
        params['v'] = False
        params['plot'] = False

    args = argparse.Namespace(**params)

    for ins in instances:
        print(ins)
        instance = tsp_to_mvctp_instance(f'../instances/{ins}')
        with open(f'/home/trolle/thesis/results/{ins[:-4]}', 'a+') as f:
            f.write('obj time\n')
            for run_number in range(args.runs):
                print(f'Run number: {run_number}')
                print(args)
                # could implement timer decorator
                st = time.process_time()
                sol = ALNS(instance, args)
                ed = time.process_time()
                f.write(f'{sol} {ed-st}\n')


