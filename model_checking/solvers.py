import subprocess


class Solver:
    def __init__(self, name, time_limit):
        self.name = name
        self.time_limit = time_limit

    def verify_properties_file(self, file_path):
        raise NotImplementedError()

    def verify_properties(self, script):
        raise NotImplementedError()


class SolverMCMAS(Solver):
    def __init__(self, path_exec, time_limit=1000*3600*4):
        self.path_exec = path_exec
        super().__init__("mcmas", time_limit)

    def verify_properties_file(self, file_path):
        result = subprocess.run([self.path_exec, file_path], capture_output=True, text=True)
        output = result.stdout  # Capturing standard output
        print(output)

        meta = self.parse_output(output)

        verification_result = meta["Formula 1"] == "TRUE"
        meta["decision"] = verification_result
        meta["status"] = "ok"
        # TODO: Handle the case of program failure or exceeding time limit
        # meta["status"] = "unknown"
        # meta["status"] = "timeout"
        return verification_result, meta

    def verify_properties(self, script):
        raise Exception("Currently SolverMCMAS handles only verification of files.")

    @staticmethod
    def parse_output(output):
        # Example output from MCMAS 1.3.0:
        # -------------------------------------------------------------------------------------------------------
        # output_x(2,2),o(1,2),x(1,1),o(0,0),x(3,1),o(2,3),x(1,3),o(3,2),x(4,0).txt has been parsed successfully.
        # Global syntax checking...
        # 1
        # 1
        # 1
        # Done
        # Encoding BDD parameters...
        # Building partial transition relation...
        # Building BDD for initial states...
        # Building reachable state space...
        # Checking formulae...
        # Verifying properties...
        #   Formula number 1: (<cross>F (crosswins && (! noughtwins))), is TRUE in the model
        #   Formula number 2: (<nought>F (noughtwins && (! crosswins))), is FALSE in the model
        # done, 2 formulae successfully read and checked
        # execution time = 3.663
        # number of reachable states = 1.01658e+07
        # BDD memory in use = 51111088
        # -------------------------------------------------------------------------------------------------------
        meta = {}
        for line in output.split('\n'):
            if "Formula number 1:" in line:
                meta["Formula 1"] = "TRUE" if "is TRUE" in line else "FALSE"
            # elif "Formula number 2:" in line:
            #     meta["Formula 2"] = "TRUE" if "is TRUE" in line else "FALSE"
            elif "execution time =" in line:
                meta["execution_time"] = float(line.split('=')[-1].strip())
            elif "number of reachable states =" in line:
                meta["reachable_states"] = float(line.split('=')[-1].strip())
            elif "BDD memory in use =" in line:
                meta["bdd_memory"] = float(line.split('=')[-1].strip())
        return meta
