import pandas as pd


def process(log_file, every=300):
    with open(log_file, "r") as f:
        lines = [x.strip() for x in f.readlines()]
        iterations = []
        train_errors = []
        train_accs = []
        valid_errors = []
        valid_accs = []
        for i, line in enumerate(lines):
            if line.startswith("Iteration "):
                iteration = int(line.split()[-1])
                if not (iteration % every == 0):
                    continue
                iterations.append(iteration)
                train_errors.append(float(lines[i + 1].split()[-1]))
                train_accs.append(float(lines[i + 2].split()[-1]))
                valid_errors.append(float(lines[i + 3].split()[-1]))
                valid_accs.append(float(lines[i + 4].split()[-1]))
            else:
                continue

        df = pd.DataFrame({"iteration": iterations,
                             "train_error": train_errors,
                             "train_accuracy": train_accs,
                             "valid_error": valid_errors,
                             "valid_accuracy": valid_accs})
        df.to_csv(log_file + ".csv", index=False)


def main():
    log_file = "curriculum.log"
    process(log_file)
    log_file = "vanilla.log"
    process(log_file)


if __name__ == "__main__":
    main()
