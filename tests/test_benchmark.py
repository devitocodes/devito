from subprocess import check_call


def run_cmd(command, problem, so, shape, nbpml, *extra):
    args = ["python", "../benchmarks/user/benchmark.py", command]
    args.extend(["-P", str(problem)])
    args.extend(["-so", str(so)])
    args.extend(["-d"] + [str(i) for i in shape])
    args.extend(["--nbpml", str(nbpml)])
    args.extend(extra)
    check_call(args)


def test_test_tti():
    run_cmd('test', 'tti', 4, [20, 20, 20], 5)


def test_test_acoustic():
    run_cmd('test', 'acoustic', 4, [20, 20, 20], 5)


def test_run_acoustic_fixed_bs():
    run_cmd('run', 'acoustic', 4, [20, 20, 20], 5, '-bs', '5', '5', '4')
